---
name: model-compute-simulation
description: "Construct model compute flow, tensor dimensions, and MFU estimation for LLM inference. Depends on model-architecture-diagram for model identification. Use when the user asks about execution flow, tensor shapes, FLOPs, MFU, operator sequence, tensor dimensions, or compute simulation for a specific model and parallelism configuration."
---

# Model Compute Simulation

## Overview

This skill constructs the model architecture, generates the operator execution sequence with tensor dimensions, and estimates MFU for LLM inference.

It depends on `model-architecture-diagram` for model name resolution and architecture diagrams. Use `model-architecture-diagram` first when the user only needs a visual architecture diagram; switch to this skill when the user needs compute-level detail.

## Confirmation Required

Before running any simulation, **ask the user to confirm** the following information. Do NOT guess or assume:

| Item | Why it matters | How to obtain | Default if user skips |
|---|---|---|---|
| Model name | Resolves to config in `model-config-index.json`; determines entire architecture | Ask user or infer from trace context | — (required) |
| Config accuracy | Indexed values may differ from actual serving config (e.g. `routed_expert_intermediate_size`, `compress_ratios`) | Ask user to provide `config.json` or verify key params against HuggingFace | Use indexed values with a caveat |
| GPU type | Determines peak FLOPS for MFU denominator | Ask user | — (required for MFU) |
| dtype (bf16 / fp8) | Affects peak FLOPS selection; fp8 doubles peak | Ask user | bf16 |
| Batch size & seq len | Directly affects FLOPs and tensor shapes | Ask user | B=1, S=1 (decode) |
| TP / DP / EP | TP splits GEMM FLOPs across GPUs; EP splits expert FLOPs | Ask user | TP=8, DP=1, EP=8 |
| Measured latency (ms) | Required for MFU numerator; must be per-GPU forward-pass wall-clock | Ask user or extract from trace via `llm-pipeline-analysis` | — (optional, no MFU without it) |

If the model is not in `model-config-index.json`, ask the user to provide a `config.json` path. Do NOT fabricate architecture parameters.

## Dependency

- `model-architecture-diagram`: resolve the model name and show the architecture diagram before drilling into compute detail.
- `llm-pipeline-analysis`: provides **compute flow** (per-kernel sequence from trace) and measured latency, which are required for per-operator MFU calculation. Use `layer_kernel_breakdown.py --format compute-flow` for human-readable compute flow and `--format json` for machine-readable export to `--kernel-flow`.

## Workflow

### Step 1: Identify the model (via model-architecture-diagram)

If the user has not already seen the architecture diagram, resolve the model name first:

```bash
python3 skills/model-architecture-diagram/scripts/model_architecture_diagram.py "<model name>"
```

Then continue with compute simulation using the same model name.

### Step 2: Construct model architecture from config

Resolve the model name and load its configuration parameters:

```bash
python3 skills/model-compute-simulation/scripts/model_compute_simulator.py "<model name>" --list-models
```

The script resolves the model name against `references/model-config-index.json`, which stores public HuggingFace config parameters (hidden_size, num_experts, MLA ranks, etc.).

If the model is not indexed, tell the user to provide a `config.json` path or request an index update.

### Step 3: Generate execution flow and tensor dimensions

Run the simulator with batch size, sequence length, and parallelism configuration:

```bash
python3 skills/model-compute-simulation/scripts/model_compute_simulator.py "Qwen3-235B-A22B" \
  --batch-size 1 --seq-len 1 \
  --tp 8 --dp 1 --ep 8 \
  --gpu h20 --dtype bf16
```

Output includes:
- Per-layer operator sequence with FLOPs and tensor shapes (shape_in → shape_out)
- Attention vs MoE/FFN FLOPs proportion per layer
- Total model FLOPs for a single forward pass

For **decode**: use `--seq-len 1`.
For **prefill**: use `--seq-len <prompt_length>`.

### Step 4: Estimate MFU with measured latency

Provide the measured forward-pass latency to compute MFU:

```bash
python3 skills/model-compute-simulation/scripts/model_compute_simulator.py "Qwen3-235B-A22B" \
  --batch-size 1 --seq-len 1 \
  --tp 8 --dp 1 --ep 8 \
  --gpu h20 --dtype bf16 \
  --measured-ms 15.0
```

MFU = theoretical_min_time / measured_time × 100%

Output includes:
- Overall MFU
- Per-layer MFU (uniform layer-time assumption)
- Per-operator FLOPs proportion (for identifying which ops dominate)

GPU peak FLOPS are loaded from `references/gpu-specs.json`. The bundled
hardware table includes H20, H100 SXM 80GB, H200 SXM 141GB, and B200 SXM
180GB. Use aliases such as `--gpu h100`, `--gpu h200`, or `--gpu b200` when
running on those local boxes.

### Step 5: Per-operator MFU with kernel-level latency (via llm-pipeline-analysis)

When you have per-kernel measured latency from `llm-pipeline-analysis`,
compute per-operator MFU by mapping kernel durations to the compute flow.

#### Method A: `--kernel-flow` (kernel-level MFU, recommended)

Use `layer_kernel_breakdown.py --format json` to produce per-kernel detail,
then feed it to the simulator for kernel-level MFU analysis.
This preserves every kernel row from the compute flow and adds FLOPs/MFU columns.

```bash
# Step 5a: Extract per-kernel detail from trace (via llm-pipeline-analysis)
python3 skills/llm-pipeline-analysis/scripts/layer_kernel_breakdown.py \
  --trace /path/to/TP-0.trace.json.gz \
  --config /path/to/config.json \
  --fwd-pass 5 --layer 3 \
  --format json > /tmp/layer3_detail.json

# Step 5b: Run simulator with kernel-flow
python3 skills/model-compute-simulation/scripts/model_compute_simulator.py "Qwen3-235B-A22B" \
  --batch-size 1 --seq-len 8192 \
  --tp 8 --dp 1 --ep 8 \
  --gpu h20 --dtype bf16 \
  --kernel-flow @/tmp/layer3_detail.json
```

The `--kernel-flow` parameter accepts a JSON string or `@file` path from
`layer_kernel_breakdown --format json`. It produces a **kernel-level MFU table**
that preserves all kernel rows from the compute flow and adds:
- `Mapped Op`: which operator this kernel maps to
- `FLOPs`: operator's total FLOPs
- `Theo(us)`: theoretical minimum time
- `MFU%`: measured FLOPs utilization
- `shape_in→shape_out`: operator tensor dimensions

When `--kernel-flow` is provided, the per-operator static template (compute flow with per-op FLOPs/shapes) is **automatically suppressed** because the kernel-level MFU table already contains per-kernel shape and FLOPs information, making the static template redundant. Only the model architecture summary, serving configuration, total model FLOPs, and kernel-level MFU table are output.

Mapping rules:
- **Direct-match kernels** (mla, moe, mhc, rmsnorm, hadamard, rope, quant, topk, etc.): time is assigned directly to the corresponding operators
- **Generic GEMM kernels** (gemm_fp8, gemm_bf16): time is distributed to remaining unassigned projection GEMM operators by FLOPs share
- **Overhead kernels** (allreduce, moe_align, moe_sort, other): rows preserved, FLOPs/MFU marked as N/A

**FP8 kernel MFU correction**: Kernels in categories `moe` (fused_moe_kernel) and `gemm_fp8` internally use fp8 precision even when `--dtype bf16` is specified. For these kernels, the MFU denominator uses the GPU's fp8 peak FLOPS (2× bf16 peak) instead of bf16 peak. The resulting MFU reflects **fp8 arithmetic utilization**, marked with a superscript `⁸` (e.g. `63.7%⁸`). This eliminates the misleading MFU > 100% that previously appeared for fused_moe kernels, and gives a true fp8 compute efficiency metric. For `gemm_bf16` kernels, the bf16 peak FLOPS is used as denominator (no superscript).

#### Method B: `--kernel-detail` (operator-level MFU, legacy)

Same input as `--kernel-flow` but outputs an **operator-level** summary table
(aggregated by operator, not per-kernel). Use when you want a compact view.

```bash
python3 skills/model-compute-simulation/scripts/model_compute_simulator.py "Qwen3-235B-A22B" \
  --batch-size 1 --seq-len 8192 \
  --tp 8 --dp 1 --ep 8 \
  --gpu h20 --dtype bf16 \
  --kernel-ms '{
    "mla": 4.922, "moe": 1.644, "allreduce": 0.769,
    "hadamard": 0.348, "mhc": 1.388, "gemm_fp8": 1.692,
    "gemm_bf16": 0.125, "rmsnorm": 0.227, "quant": 0.311,
    "rope": 0.209, "topk": 0.122, "activation": 0.071,
    "other": 0.437
  }'
```

The `--kernel-ms` parameter accepts a JSON object mapping **kernel category** names
to their measured durations in **milliseconds**. It uses FLOPs-proportional
distribution across entire categories, which is less precise than `--kernel-detail`
because generic GEMM categories (gemm_fp8, gemm_bf16) span multiple operator categories.

Output includes:
- Model architecture summary (layers, hidden_size, attention_type, MoE config)
- Per-layer compute flow: operator sequence with tensor dimensions, FLOPs, shape_in→shape_out
- Per-operator MFU table: each operator's FLOPs, theoretical time, measured time (from trace), MFU%
- Kernel → operator mapping explanation (direct-match vs FLOPs-proportional vs overhead)
- Overall and per-layer MFU

## When To Use It

- after viewing a model architecture diagram, when you need compute-level detail
- when the user asks about execution flow, tensor dimensions, or FLOPs for a specific serving shape
- when the user asks about MFU and can provide measured forward-pass latency
- when comparing compute profiles across different parallelism configurations

## Useful Commands

List known model IDs:

```bash
python3 skills/model-compute-simulation/scripts/model_compute_simulator.py --list-models
```

List known GPU types:

```bash
python3 skills/model-compute-simulation/scripts/model_compute_simulator.py --list-gpus
```

Emit JSON for automation:

```bash
python3 skills/model-compute-simulation/scripts/model_compute_simulator.py "GLM-5" --format json
```

## Output Contract

Return:

1. **Model architecture summary**: model name, config source, num_layers, hidden_size, attention_type, MoE config (num_experts, topk, shared_experts), MHC, head_dim
2. **Serving configuration**: batch_size, seq_len, TP, DP, EP, GPU, dtype
3. **Per-layer compute flow** (showing first representative layer in detail):
   - Operator sequence table: name, category, FLOPs, shape_in → shape_out
   - Attention vs MoE/FFN FLOPs proportion
4. **Total model FLOPs** for a single forward pass
5. **Kernel-level MFU table** (when `--kernel-flow` provided) — builds on `llm-pipeline-analysis` compute flow:
   - Preserves ALL kernel rows from the compute flow (never deleted)
   - Per-kernel columns: `# | Half | Category | Simplified Name | dur(us) | % | Mapped Op | FLOPs | Theo(us) | MFU% | shape_in→shape_out`
   - Direct-match kernels: show mapped operator FLOPs/MFU
   - Overhead kernels: show N/A for FLOPs/MFU, row preserved
6. **Operator-level MFU table** (when `--kernel-detail` or `--measured-ms` provided):
   - Each operator: name, category, total FLOPs, per-GPU FLOPs, theoretical time, measured time (from trace), MFU%
   - Kernel category → operator mapping explained
7. **Overall MFU** and **per-layer MFU**
8. **One-line summary**: dominant compute category, MFU status, key bottleneck

## Trace-Based Validation (extract_compute_flow_from_trace.py)

Use `scripts/extract_compute_flow_from_trace.py` to extract the real operator sequence and tensor dimensions from a torch profiler trace, then compare against the static template as ground truth validation.

```bash
# Extract compute flow from a trace
python3 skills/model-compute-simulation/scripts/extract_compute_flow_from_trace.py \
  --input /path/to/trace.json.gz --format text

# Compare trace against static template
python3 skills/model-compute-simulation/scripts/extract_compute_flow_from_trace.py \
  --input /path/to/trace.json.gz \
  --compare qwen3-235b-a22b \
  --batch-size 1 --seq-len 1 --tp 8 --ep 8
```

### Compute Flow Confirmation Hierarchy

When the static template or trace extraction cannot fully confirm the compute process (e.g. ambiguous scope, missing shapes, new model architecture), follow this escalation hierarchy:

1. **Static template** (`model_compute_simulator.py` + `model-config-index.json`) — fast, covers known models
2. **Trace extraction** (`extract_compute_flow_from_trace.py`) — validates template against real execution
3. **Inference framework source code** — when trace is insufficient (missing `Input Dims`, CUDA Graph replay, compiled kernels without scope), read the model's forward flow directly from the serving framework source:
   - **SGLang**: `python/sglang/srt/models/<model_name>.py` — contains the `forward()` method with the exact operator sequence, tensor shapes, and parallelism split logic
   - **vLLM**: `vllm/model_executor/models/<model_name>.py`
   - **TensorRT-LLM**: `cpp/tensorrt_llm/pyexecutor/py_executor.cpp` + model config files

   When consulting framework source, focus on:
   - The `forward()` method: operator call order and residual connections
   - QKV / O projection: whether LoRA-style down/up projections are used (`q_lora_rank`, `o_lora_rank`)
   - MoE routing: top-k selection, shared vs routed expert split
   - TP/EP slicing: which dimensions are split and how FLOPs divide across GPUs
   - Any model-specific ops not in the static template (e.g. MHC, Hadamard, indexer)

   **Action**: If the framework source reveals discrepancies with the static template, update `model-config-index.json` and/or `build_layer_ops()` accordingly.

### Limitations of Trace Extraction

| Limitation | Detail | Workaround |
|---|---|---|
| `record_shapes=True` required | Trace must be captured with shape recording enabled; without it, `Input Dims` fields are absent and FLOPs cannot be computed | SGLang live capture and vLLM `torch_profiler_with_stack=true` already enable this; TensorRT-LLM requires a `py_executor.py` override adding `record_shapes=True` |
| CUDA Graph mode | During graph replay, `cpu_op` events may only appear once (at capture time); shape information for replayed iterations is not re-recorded | The script detects graph capture phases and annotates affected ops; use eager-mode traces for full coverage |
| TP-sliced dimensions | Trace shows post-TP-split dimensions (e.g. `H/TP`), not the full-model view | Use `--tp` in `--compare` mode to scale trace FLOPs back to full-model equivalents |
| Scope attribution quality | Python scope depends on `with_stack=True`; some frameworks or compiled paths may produce shallow or missing scope chains | Graceful degradation: ops with unresolved scope are categorized as "other" |
| Not a replacement for static templates | Trace extraction is a validation and discovery tool; static templates remain the primary fast-analysis path | Use trace extraction to verify templates for new models, then update `model-config-index.json` if discrepancies are found |

## References

- `references/model-config-index.json`: model configuration parameters (hidden_size, expert counts, MLA ranks, etc.).
- `references/gpu-specs.json`: GPU peak FLOPS specifications for MFU calculation.
- `scripts/extract_compute_flow_from_trace.py`: trace-based compute flow extraction and template validation tool.
