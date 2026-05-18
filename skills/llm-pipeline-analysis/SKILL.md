---
name: llm-pipeline-analysis
description: "Layer-level pipeline analysis for LLM torch profiler traces. Decomposes a forward pass into per-layer wall-clock and sum-duration, classifies layers by model architecture (compress_ratio, hash layers), identifies bottleneck layers, and maps kernels to computation categories. Use after triage to drill into per-layer and per-kernel detail."
---

# LLM Pipeline Analysis

## Overview

This skill decomposes an LLM `torch.profiler` trace into **per-layer** and **per-kernel** detail, answering:

- Which layers dominate the forward pass?
- How do layer types (C4 LIGHT, C128 HEAVY, HASH, FULL_ATTN) differ?
- What is the kernel-level breakdown inside a specific layer?
- Where in the Perfetto UI timeline does each layer sit?

It is a **complement** to `llm-torch-profiler-analysis` (triage):
triage gives the whole-trace top-k kernel table; this skill drills into
per-layer structure and identifies which layer types are the bottleneck.

## When To Use It

- after triage, when you need to know **which layers** contribute most
- when the model has alternating layer types (e.g. models with
  `compress_ratios` like DeepSeek-V4 NSA)
- when you need to compare cold-start vs steady-state forward passes
- when you need to navigate to a specific layer in Perfetto UI
- when you need to select representative layers for deep-dive analysis

## Confirmation Required

Before running any script, **ask the user to confirm** the following information. Do NOT guess or assume:

| Item | Why it matters | How to obtain | Default if user skips |
|---|---|---|---|
| Model name | Determines which `config.json` to use; affects layer classification | Ask user | — (required) |
| Model profile | Determines anchor kernel, blocks-per-layer, and kernel classification rules | Ask user or auto-infer from config | Auto-inferred from config |
| `config.json` path | Provides `compress_ratios`, `num_hidden_layers`, `num_hash_layers` etc. | Ask user or search filesystem | — (required) |
| GPU type | Needed for MFU cross-reference with `model-compute-simulation` | Ask user | — |
| TP / EP | Parallelism config affects kernel naming and AllReduce count | Ask user or infer from trace filename (e.g. `TP-0`) | TP=8, EP=8 |
| Serving mode | Decode vs prefill changes kernel mix and FLOPs profile | Ask user | decode B=1 |

If the user cannot provide `config.json`, search for it in common locations (`/root/workspace/*/config.json`, HuggingFace cache). If still not found, the user must specify `--profile` explicitly.

## Model Profiles

Scripts use **ModelProfile** to determine layer boundary detection and kernel
classification. Profiles are auto-inferred from `config.json` or selected
via `--profile`:

| Profile | Anchor kernel | Blocks/layer | Layer structure | Auto-infer condition |
|---|---|---|---|---|
| `dsv4_csa_hca` | `mhc_post_tilelang` | 2 | attn + ffn halves | `compress_ratios` non-empty |
| `dsv3_mla` | `flash_fwd_mla_combine` | 1 | full layer | `kv_lora_rank > 0` |
| `generic` | auto-detect or `--anchor-kernel` | 1 | full layer | fallback |

Use `--profile generic --anchor-kernel YOUR_KERNEL` for models not covered
by built-in profiles.

## Prerequisites

- A `torch.profiler` trace in Chrome-trace JSON format (`.json` or `.json.gz`)
- The model's `config.json` (for profile inference, `compress_ratios`, etc.)
- The trace must contain a recognizable layer-boundary anchor kernel
  (auto-detected from the profile, or specified via `--anchor-kernel`)

## Layer Boundary Detection

The scripts use an anchor kernel as a layer-boundary marker. The anchor and
layer structure are determined by the active **ModelProfile**.

For example, with the `dsv4_csa_hca` profile, each transformer layer produces
**2 consecutive** `mhc_post_tilelang` calls:

```
mhc_post_tilelang  ← end of attn half (attention + O-proj + AllReduce)
  ... ffn computation ...
mhc_post_tilelang  ← end of ffn half (MoE experts + AllReduce)
  ... next layer attn ...
mhc_post_tilelang  ← next layer's attn boundary
```

So for N layers with the `dsv4_csa_hca` profile, one forward pass has `2N`
anchor blocks.  With `dsv3_mla` or `generic`, each layer has 1 block.

Forward pass `P` starts at block index `P * (N * blocks_per_layer)`.

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

# Auto-select first steady-state pass
python3 scripts/layer_timeline_analyzer.py \
  --trace /path/to/TP-0.trace.json.gz \
  --config /path/to/config.json
```

Output includes:
- Per-layer wall-clock time, sum-duration, and category breakdown (MLA, MoE, GEMM, NCCL, MHC, Hadamard)
- Layer cluster statistics grouped by type (C4_LIGHT, C128_HEAVY, HASH, etc.)
- All-passes summary showing cold-start → steady-state growth

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

# JSON export for model-compute-simulation
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
- `--format text` (default): grouped summary + ordered kernel list with simplified names and durations
- `--format compute-flow`: model architecture summary + category-level timing + per-kernel table with `Category` column; this is the bridge format for `model-compute-simulation`
- `--format json`: machine-readable per-kernel detail for `model-compute-simulation --kernel-flow`
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

Output includes:
- Forward pass time ranges in Perfetto-relative seconds
- Per-layer start/end times with compress_ratio labels

## Workflow

### Step 1: Identify steady-state forward pass

```bash
python3 scripts/layer_timeline_analyzer.py \
  --trace $TRACE --config $CONFIG --show-all-passes
```

Read the "all-passes" table. The first pass is cold-start (few tokens).
Find the first pass where layer-0 wall-clock stabilizes (typically pass 3-5).

### Step 2: Per-layer breakdown on steady-state pass

```bash
python3 scripts/layer_timeline_analyzer.py \
  --trace $TRACE --config $CONFIG --fwd-pass 5
```

Identify:
- Which layer type dominates (C4_LIGHT vs C128_HEAVY vs HASH)
- The MLA / MoE / GEMM / NCCL proportion per layer type
- Which layer type has the most optimization headroom

### Step 3: Compute flow for representative layer(s)

Select 1-2 representative layers (one per bottleneck type), then:

```bash
# Human-readable compute flow table
python3 scripts/layer_kernel_breakdown.py \
  --trace $TRACE --config $CONFIG \
  --fwd-pass 5 --layer 3 --format compute-flow

# JSON export for model-compute-simulation
python3 scripts/layer_kernel_breakdown.py \
  --trace $TRACE --config $CONFIG \
  --fwd-pass 5 --layer 3 --format json > /tmp/layer3_detail.json
```

The `--format compute-flow` output includes:
- Model architecture summary at the top
- Category-level timing summary
- Per-kernel table with `# | Half | Category | Simplified Name | dur(us) | %`

Pass the JSON export to `model-compute-simulation` via `--kernel-flow` for per-operator MFU analysis.

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
| `compress_ratios[i]` | 0 | FULL_ATTN | No NSA compression (layers 0-1) |
| `compress_ratios[i]` | 4 | C4_LIGHT | C128 sparse attention, fastest |
| `compress_ratios[i]` | 128 | C128_HEAVY | C4 attention + Hadamard + Indexer, bottleneck |
| `i >= N - num_hash_layers` | — | HASH | Hash-table routing with paged MQA |
| `i == 0` | — | FIRST | First layer (empty KV cache) |
| `i == N - 1` | — | FINAL | Final layer (lm_head output) |

## Kernel Categories

Kernels are classified by the active ModelProfile's rules. Categories marked
with (DSv4) are specific to the `dsv4_csa_hca` profile; all profiles include
the universal categories.

| Category | Match Pattern | Profile | Typical Share (DSv4) |
|---|---|---|---|
| ★ MLA Attention | `flash_fwd_splitkv_mla` | DSv4, DSv3 | 21-33% |
| ★ MoE Fused | `fused_moe_kernel` | DSv4, DSv3 | 11-17% |
| ● NCCL AllReduce | `AllReduce` | universal | 5-8% |
| GEMM fp8 | `deep_gemm` | universal | 12-25% |
| GEMM bf16 | `nvjet` | universal | 11-13% |
| Hadamard Xform | `hadamard` | DSv4 | 0-2.4% |
| Indexer Cache | `indexer` | DSv4 | 0-0.1% |
| Paged MQA | `paged_mqa_logits` | DSv4 | 0-1.8% |
| MHC | `mhc_pre_gemm_sqrsum`, `mhc_pre_big_fuse`, `mhc_post_tilelang` | DSv4 | 10-15% |
| C4/C128 Prefill | `c4_prefill`, `c128_prefill` | DSv4 | 0-0.3% |
| RMSNorm | `RMSNorm`, `rms_normalize` | universal | 1-2% |
| FP8 Quant | `quant`, `Quant` | universal | 1-2% |
| TopK | `topk` | universal | 0-0.7% |
| RoPE | `deepseek_rope`, `fused_norm_rope` | DSv4, DSv3 | 1-2% |
| Activation | `silu_mul_clamp`, `act_and_mul` | universal | 0-0.5% |
| Other | — | universal | 2-5% |

## Output Contract

Return:

1. **Trace metadata**: trace path, model config path, GPU type, TP/EP
2. **Model Architecture Summary** (from `config.json`):
   - model name, num_layers, hidden_size, num_attention_heads, num_key_value_heads, head_dim
   - Attention type (e.g. csa_hca), Q/O LoRA ranks
   - MoE config: num_experts, topk, num_shared_experts, intermediate_size
   - MHC config (if applicable)
   - NSA config (if applicable): index_n_heads, index_head_dim, index_topk, qk_rope_head_dim, sliding_window
   - compress_ratios distribution (how many C4_LIGHT / C128_HEAVY / FULL_ATTN / HASH layers)
3. **Per-batch forward passes summary table** (from `layer_timeline_analyzer.py --show-all-passes`):
   - Columns: Fwd#, Start(s), End(s), Duration(ms), Avg Layer(ms), First Layer(ms), Notes
   - Identifies cold-start vs steady-state passes
4. **Chosen forward pass**: index and rationale (cold-start vs steady-state)
5. **Per-layer wall-clock and sum-duration table** (from `layer_timeline_analyzer.py --fwd-pass N`):
   - Columns: L, c_r, Type, Wall(ms), SumDur(ms), MLA, MoE, GEMM, NCCL, MHC, Hadam, AR#, K#
   - Each row is one layer, with layer type label
6. **Layer cluster statistics table** grouped by type:
   - Columns: Cluster, #, Avg Wall(ms), Avg Sum(ms), MLA%, MoE%, GEMM%, NCCL%, MHC%, Hadam%
   - Identifies bottleneck layer type and optimization headroom
7. **Compute Flow Table** for selected representative layer(s) — this is the bridge to `model-compute-simulation`:
   - Produced by `layer_kernel_breakdown.py --format compute-flow`
   - Columns: `# | Half | Category | Simplified Name | dur(us) | %`
   - Category-level summary above the table
   - Also export as JSON (`--format json`) for `model-compute-simulation` consumption
8. **Perfetto UI time ranges** when requested
9. **One-line summary**: bottleneck layer type and top optimization target