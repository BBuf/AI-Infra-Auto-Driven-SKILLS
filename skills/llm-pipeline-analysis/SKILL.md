---
name: llm-pipeline-analysis
description: "Inspect LLM torch profiler traces at forward-pass, layer, and kernel level. Use when you need layer timings, anchor-kernel boundaries, representative kernel flows, or Perfetto time ranges."
---

# LLM Pipeline Analysis

## Overview

Use this when a whole-trace profiler summary is too coarse. The scripts read a
Chrome-trace JSON file, find layer-boundary anchor kernels, group kernels into
forward passes and layers, and print timing tables you can use for Perfetto
navigation or detailed timing analysis.

## When To Use It

- when you need to know **which layers** contribute most
- when the model has alternating layer types (e.g. models with
  `compress_ratios` like DeepSeek-V4 NSA, or hybrid GDN/GQA stacks such as
  Qwen3.8-27B)
- when you need to compare cold-start vs steady-state forward passes
- when you need to navigate to a specific layer in Perfetto UI
- when you need to select representative layers for deep-dive analysis

## Verify inputs from artifacts

Resolve model config, phase, rank/device and TP/EP/DP from the run manifest,
server arguments and trace. Ask only for missing information that changes the
analysis. `TP-0` identifies a rank, not TP world size; do not default an unknown
run to TP8/EP8 or decode BS1. Speculative request BS and target-verify M differ.
If no config is available, use an explicit profile and verified layer count.

## Model Profiles

Scripts use **ModelProfile** to determine layer boundary detection and kernel
classification. Profiles are auto-inferred from `config.json` or selected
via `--profile`:

| Profile | Anchor kernel | Blocks/layer | Auto-infer condition |
|---|---|---|---|
| `dsv4_csa_hca` | `mhc_pre_big_fuse` | 2 | DeepSeek-V4 or compression fallback |
| `dsv41` | explicit verified anchor | 1 (override for sublayers) | V4.1 text/wrapper identity or KV-source layers |
| `dsv3_mla` | explicit backend anchor or residual boundary | backend-dependent | homogeneous MLA after identity dispatch |
| `mla_dsa` | verified residual boundary | 2 | GLM-5/5.2/5.3, V3.2, DSA MLA |
| `kimi_k3` | explicit `attn_res_fused` (SGLang) / `attn_res_fwd_online_v2_kernel` (vLLM) | 2 + one output aggregation/pass | K3 / Kimi linear with attention residual |
| `glm5_next` | `mhc_pre_big_fuse` | 2 | GLM-5.3-Flash |
| `hy_v4` | explicit `ihc_pre_stage2` on Triton fallback | 2 | Hy4 (hpc symbol requires trace check) |
| `qwen4_exp` | explicit `grouped_gemma_rmsnorm` | 2 | Qwen3.8-Flash-Next (vLLM may fuse norm) |
| `nemotron_h` | explicit residual norm | 1 | one Mamba/attention/MoE block per config layer |
| `longcat_flash` | explicit residual boundary | 4 | LongCat-Flash architecture |
| `gemma4` | explicit `_gemma_qkv_rmsnorm_kernel` (SGLang CUDA) | 1 | Gemma4 text/wrapper identity |
| `generic_addnorm` | verified fused-add RMSNorm or allreduce fusion | 2 | other configured models |
| `generic` | explicit anchor; residual candidates only | 1, or 2 for residual candidates | config-less fallback |

Nested `text_config`, `language_config` and `llm_config` are flattened before
inference. Model identity takes priority over `kv_lora_rank`: K3, GLM-5 Next,
Hy4, dots3 and LongCat must not be treated as homogeneous V3 MLA.
Read layer count from config: V4 Flash has 43 layers, Pro 61, V4.1 40.
These anchor families are source-verified for SGLang/vLLM, with backend and
phase caveats; TensorRT-LLM/TokenSpeed cadence requires separate trace evidence.

## Prerequisites

- A `torch.profiler` trace in Chrome-trace JSON format (`.json` or `.json.gz`)
- The model's `config.json` (for profile inference, `compress_ratios`, etc.)
- The trace must contain a recognizable layer-boundary anchor kernel
  (auto-detected from the profile, or specified via `--anchor-kernel`)

## Layer Boundary Detection

The scripts use an anchor kernel as a layer-boundary marker. The anchor and
layer structure are determined by the active **ModelProfile**.

Find the model's residual kind at its stage boundary first. SGLang declares
attention/FFN boundaries through `python/sglang/srt/layers/layer_boundary/factories.py`
(add-norm, gated, iHC, mHC, post-norm and stream). Backend dispatch determines
the visible boundary kernel.

V4 currently starts each half at `mhc_pre_big_fuse*`: attention-pre then
FFN-pre, two anchors per layer. Fused post+pre removes `mhc_post_tilelang`
for small-token SGLang traces; retain that old anchor only for verified unfused
traces or the >32-token path. V4.1 uses previous-sublayer pre-mix and disables
that cross-layer fused path, so it always requires explicit anchor evidence.
FlashMLA's `flash_fwd_mla_combine` depends on backend and split count; SM100
SGLang V3 defaults to TRT-LLM MLA, while DSA models use DSA.

All scripts accept `--blocks-per-layer K --anchor-offset I --pid PID --device D`.
`I` is a verified first-layer anchor in the selected GPU, not inferred from
counts. Mixed ranks/devices and a count residue modulo `N*K` are rejected.
A divisible count is necessary but insufficient: separate target/draft phases,
MTP/DSPARK passes and TBO microbatches first. K3's extra output aggregation
requires a phase-filtered trace for these timing analyzers; the layer-track
helper supports the extra anchor explicitly. The final selected pass ends at
the selected GPU event boundary, which must itself be verified as a complete
pass. Anchors are navigation intervals; cross-stream kernels may overlap them.

For compiled vLLM traces, norms can lower to Inductor kernels. Capture a short
mapping trace with explicit norm custom ops or eager execution and verify an
attention/backend anchor on the actual measured trace.

Source audit 2026-10-05: [SGLang V4 dispatch](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/srt/models/deepseek_v4.py),
[mHC kernels](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/kernels/ops/layernorm/mhc.py),
[vLLM mHC dispatch](https://github.com/vllm-project/vllm/blob/0c16eee3f1ff777298cc894c3eeb85f3880c6d6a/vllm/model_executor/kernels/mhc/tilelang.py).

## Scripts

### 1. `layer_timeline_analyzer.py` — Per-layer timeline and cluster stats

```bash
# Show all forward passes summary (cold-start vs steady-state)
python3 scripts/layer_timeline_analyzer.py \
  --trace /path/to/TP-0.trace.json.gz \
  --config /path/to/config.json \
  --show-all-passes

# Detailed per-layer breakdown for a specific forward pass
python3 scripts/layer_timeline_analyzer.py \
  --trace /path/to/TP-0.trace.json.gz \
  --config /path/to/config.json \
  --fwd-pass 5

# Auto-select the first relatively stable pass window
python3 scripts/layer_timeline_analyzer.py \
  --trace /path/to/TP-0.trace.json.gz \
  --config /path/to/config.json
```

The script prints:
- Per-layer wall-clock time, sum-duration, and category breakdown (MLA, MoE, GEMM, NCCL, MHC, Hadamard)
- Layer cluster statistics grouped by type (CSA_C4, HCA_C128, HASH, etc.)
- All-passes summary showing cold-start → steady-state growth

Automatic steady-state selection requires two consecutive layer-0 timing
changes within 5%. It does not use an absolute latency threshold, so the same
rule works across model sizes and accelerators. If no stable window exists,
choose `--fwd-pass` explicitly.

### 2. `layer_kernel_breakdown.py` — Per-layer kernel detail and compute flow

```bash
# Single layer kernel dump
python3 scripts/layer_kernel_breakdown.py \
  --trace /path/to/TP-0.trace.json.gz \
  --config /path/to/config.json \
  --fwd-pass 5 --layer 3

# Compute flow format (with model architecture summary and category column)
python3 scripts/layer_kernel_breakdown.py \
  --trace /path/to/TP-0.trace.json.gz \
  --config /path/to/config.json \
  --fwd-pass 5 --layer 3 --format compute-flow

# JSON export
python3 scripts/layer_kernel_breakdown.py \
  --trace /path/to/TP-0.trace.json.gz \
  --config /path/to/config.json \
  --fwd-pass 5 --layer 3 --format json

# Compare two layers side-by-side
python3 scripts/layer_kernel_breakdown.py \
  --trace /path/to/TP-0.trace.json.gz \
  --config /path/to/config.json \
  --fwd-pass 5 --layer 2 --compare-layer 3
```

Output formats:
- `--format text` (default): grouped summary + top hot kernels ranked by duration, with simplified names and percentages
- `--format compute-flow`: model architecture summary + per-kernel hotness table with `Category`, `%`, and `ts_rel(ms)` columns
- `--format json`: one machine-readable JSON document ranked by duration; with
  `--compare-layer`, it contains `primary`, `comparison`, and `kernel_diff`
- Kernel diff when comparing two layers (unique kernels in each)

### 3. `perfetto_time_mapper.py` — Perfetto UI time navigation

```bash
# Show all forward pass time ranges in Perfetto
python3 scripts/perfetto_time_mapper.py \
  --trace /path/to/TP-0.trace.json.gz \
  --config /path/to/config.json

# Layer-level time ranges for a specific forward pass
python3 scripts/perfetto_time_mapper.py \
  --trace /path/to/TP-0.trace.json.gz \
  --config /path/to/config.json \
  --fwd-pass 5 --layers 2,3,38,42
```

The script prints:
- Forward pass time ranges in Perfetto-relative seconds
- Per-layer start/end times with compress_ratio labels

## Workflow

### Step 1: Identify steady-state forward pass

```bash
python3 scripts/layer_timeline_analyzer.py \
  --trace $TRACE --config $CONFIG --show-all-passes
```

Read the "all-passes" table. The first pass is cold-start (few tokens).
Use the first relative-stability window selected by the script, or choose a
pass explicitly when timings continue to change.

### Step 2: Per-layer breakdown on steady-state pass

```bash
python3 scripts/layer_timeline_analyzer.py \
  --trace $TRACE --config $CONFIG --fwd-pass 5
```

Identify:
- Which layer type dominates (CSA_C4 vs HCA_C128 vs HASH)
- The MLA / MoE / GEMM / NCCL proportion per layer type
- Which layer type is the best next target

### Step 3: Compute flow for representative layer(s)

Select 1-2 representative layers (one per bottleneck type), then:

```bash
# Human-readable compute flow table
python3 scripts/layer_kernel_breakdown.py \
  --trace $TRACE --config $CONFIG \
  --fwd-pass 5 --layer 3 --format compute-flow

# JSON export
python3 scripts/layer_kernel_breakdown.py \
  --trace $TRACE --config $CONFIG \
  --fwd-pass 5 --layer 3 --format json > /tmp/layer3_detail.json
```

The `--format compute-flow` output includes:
- Model architecture summary at the top
- Per-kernel hotness table with `# | Half | Category | Simplified Name | dur(us) | % | ts_rel(ms) | Input Dims`
- Rows are ranked by `dur(us)` descending by default; use `ts_rel(ms)` to jump back to the kernel's trace location.

### Step 4: Compare layer types (optional)

```bash
python3 scripts/layer_kernel_breakdown.py \
  --trace $TRACE --config $CONFIG \
  --fwd-pass 5 --layer 2 --compare-layer 3
```

This shows the exact kernel difference between the two layer types.

### Step 5: Navigate in Perfetto UI (optional)

```bash
python3 scripts/perfetto_time_mapper.py \
  --trace $TRACE --config $CONFIG \
  --fwd-pass 5 --layers 2,3,38,42
```

Use the printed time ranges to navigate directly in Perfetto.

## Layer Type Classification

The scripts classify layers based on `config.json` fields:

| Config field | Value | Layer Type | Description |
|---|---|---|---|
| `compress_ratios[i]` | 0 | SWA_ONLY | window-only attention, no compressor/indexer |
| `compress_ratios[i]` | 4 | CSA_C4 | compressor + lightning indexer (Hadamard/paged-MQA/top-k) + sparse attention |
| `compress_ratios[i]` | 128 | HCA_C128 | compressor + all compressed rows and window; no indexer |
| `compress_ratios[i]` | 2 / 1 | V41_C2 / V41_C1 | source-layer compressor/indexer; other layers share KV/index |
| `i < num_hash_layers` | — | +HASH | hash routing in the FIRST N layers, composable with FIRST/FINAL |
| `i == 0 / N-1` | — | +FIRST / +FINAL | positional flags, never replace attention/routing work |

V4 Flash/Pro hash-route L0–2. V4.1 has no hash layers; its KV source layers
are `[2,8,14,20]`, index sources `[2,8,14,20,24,28,32,36]`, candidate source
20, and Engram layers `[1,14]`. Use these role fields when choosing representative
layers. Pro starts with ratio-128 layers; do not assume its first two are SWA.
K3's KDA layer ids are 1-indexed, whereas GLM-5 Next uses 0-indexed KDA ids.

## Kernel Categories

Kernels are classified by the active ModelProfile's rules. Categories marked
with (DSv4) are specific to the `dsv4_csa_hca` profile; all profiles include
the universal categories.

| Category | Match Pattern | Profile | Typical Share (DSv4) |
|---|---|---|---|
| ★ MLA Attention | `flash_fwd_splitkv_mla` | DSv4, DSv3 | 21-33% |
| ★ MoE Fused | `fused_moe_kernel` | DSv4, DSv3 | 11-17% |
| ● NCCL AllReduce | case-insensitive `allreduce`, `all_reduce`, `cross_device_reduce`, `reduce_scatter`, NCCL/Lamport | universal | 5-8% |
| GEMM fp8 | `deep_gemm` | universal | 12-25% |
| GEMM bf16 | `nvjet` | universal | 11-13% |
| Hadamard Xform | `hadamard` | DSv4 | 0-2.4% |
| Indexer Cache | `indexer` | DSv4 | 0-0.1% |
| Paged MQA | `paged_mqa_logits` | DSv4 | 0-1.8% |
| MHC | `mhc_pre_gemm_sqrsum`, `mhc_pre_big_fuse`, `mhc_post_tilelang` | DSv4 | 10-15% |
| C4/C128 Prefill | `c4_prefill`, `c128_prefill` | DSv4 | 0-0.3% |
| RMSNorm | case-insensitive `rms_?norm`, `layernorm` | universal | 1-2% |
| FP8 Quant | `quant`, `Quant` | universal | 1-2% |
| TopK | `topk` | universal | 0-0.7% |
| RoPE | `deepseek_rope`, `fused_norm_rope` | DSv4, DSv3 | 1-2% |
| Activation | `silu_mul_clamp`, `act_and_mul` | universal | 0-0.5% |
| Other | — | universal | 2-5% |

## Reporting Checklist

Include:

1. **Trace metadata**: trace path, model config path, GPU type, TP/EP
2. **Model Architecture Summary** (from `config.json`):
   - model name, num_layers, hidden_size, num_attention_heads, num_key_value_heads, head_dim
   - Attention type (e.g. csa_hca), Q/O LoRA ranks
   - MoE config: num_experts, topk, num_shared_experts, intermediate_size
   - MHC config (if applicable)
   - NSA config (if applicable): index_n_heads, index_head_dim, index_topk, qk_rope_head_dim, sliding_window
   - compress_ratios distribution (how many CSA_C4 / HCA_C128 / SWA_ONLY / HASH layers)
3. **Per-batch forward passes summary table** (from `layer_timeline_analyzer.py --show-all-passes`):
   - Columns: Fwd#, Start(s), End(s), Duration(ms), Avg Layer(ms), First Layer(ms), Notes
   - Identifies cold-start vs steady-state passes
4. **Chosen forward pass**: index and rationale (cold-start vs steady-state)
5. **Per-layer wall-clock and sum-duration table** (from `layer_timeline_analyzer.py --fwd-pass N`):
   - Columns: L, c_r, Type, Wall(ms), SumDur(ms), MLA, MoE, GEMM, NCCL, MHC, Hadam, AR#, K#
   - Each row is one layer, with layer type label
6. **Layer cluster statistics table** grouped by type:
   - Columns: Cluster, #, Avg Wall(ms), Avg Sum(ms), MLA%, MoE%, GEMM%, NCCL%, MHC%, Hadam%
   - Identifies bottleneck layer type and likely next target
7. **Compute Flow Table** for selected representative layer(s):
   - Produced by `layer_kernel_breakdown.py --format compute-flow`
   - Columns: `# | Half | Category | Simplified Name | dur(us) | % | ts_rel(ms) | Input Dims`
   - Rows are sorted by top hot kernels (`dur(us)` descending) by default
   - Optional JSON export (`--format json`)
8. **Perfetto UI time ranges** when requested
9. **One-line summary**: bottleneck layer type and likely next target
