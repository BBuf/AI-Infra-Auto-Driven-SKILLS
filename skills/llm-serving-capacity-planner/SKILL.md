---
name: llm-serving-capacity-planner
description: "Analyze GPU memory decomposition and estimate max concurrent requests from LLM serving logs. Use when the user asks about GPU memory decomposition, memory analysis, KV cache capacity, max concurrency, capacity planning, mem-fraction-static, or memory budget for LLM inference."
---

# LLM Serving Capacity Planner

## Overview

This skill parses LLM serving framework (SGLang/vLLM) startup logs, automatically extracts memory-related data points, decomposes GPU memory into categories (weights, KV cache pool, CUDA graph, framework overhead), and estimates max concurrent requests under different token-length scenarios.

It complements `model-compute-simulation` (which focuses on FLOPs/MFU) by addressing the **capacity planning** dimension: how much GPU memory each component consumes and how many requests can be served concurrently.

## Confirmation Required

Before running any analysis, **ask the user to confirm** the following information. Do NOT guess or assume:

| Item | Why it matters | How to obtain | Default if user skips |
|---|---|---|---|
| Log file path | Primary input; all memory data comes from here | Ask user for the serving startup log | — (required) |
| GPU type | Determines total HBM for decomposition validation | Ask user or infer from log | Auto-detected from log if possible |
| nvidia-smi output | Provides per-rank actual memory for cross-validation | Capture with `nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader > smi.txt` | — (optional, but recommended) |
| Model config.json | Enables theoretical KV cache byte calculation and replication factor analysis | Ask user for the model's config.json path | — (optional, log data used instead) |
| Request token length | Determines concurrency estimate denominator | Ask user | 4096, 6144, 8192 |

## Dependency

- `model-compute-simulation/references/gpu-specs.json`: GPU HBM specifications (read-only, shared reference). Known aliases include `h20`, `h100`, `h200`, and `b200`.
- `llm-serving-auto-benchmark`: can capture startup logs and nvidia-smi data as part of benchmark setup.

## Workflow

### Step 1: Collect the serving log

The user should provide the startup log from an SGLang or vLLM serving instance. Key log lines that the analyzer needs:

- `Load weight begin. avail mem=XX GB`
- `Memory profiling: available_gpu_memory=XX GB, ...` (newer sglang)
- `SW KV memory calculation: bytes_per_full_token=XX, available_bytes=XX GB, full_token=XX` (SWA models like DeepSeek-V4)
- `Memory pool end. avail mem=XX GB`
- `Capture cuda graph end. ... mem usage=XX GB. avail mem=XX GB.`
- `max_total_num_tokens=XX, ... max_running_requests=XX, ... available_gpu_mem=XX GB`
- `server_args=ServerArgs(...)` (for serving parameters)

If the log is from a running instance, capture it by redirecting stdout/stderr to a file at launch time.

### Step 2: Optionally capture nvidia-smi data

For per-rank memory comparison:

```bash
docker exec <container> nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader > smi.txt
```

### Step 3: Run the analyzer

```bash
python3 skills/llm-serving-capacity-planner/scripts/capacity_analyzer.py \
  --log-file /path/to/sglang.log \
  --nvidia-smi-file /path/to/smi.txt \
  --gpu h200 \
  --config-json /path/to/config.json
```

For JSON output (automation):

```bash
python3 skills/llm-serving-capacity-planner/scripts/capacity_analyzer.py \
  --log-file /path/to/sglang.log \
  --format json
```

### Step 4: Review and interpret results

The analyzer outputs:

1. **Memory breakdown table**: each category (weights, KV pool, CUDA graph, framework, other) with GiB, MiB, percentage, and derivation
2. **Per-rank comparison**: nvidia-smi data across all TP ranks
3. **KV pool detail**: pool configuration, KV dtype, replication factor, per-token byte calculation
4. **Concurrency estimate**: max concurrent requests for different token lengths
5. **Tuning suggestions**: actionable recommendations for increasing capacity

## When To Use It

- After launching an LLM serving instance, to understand how GPU memory is distributed
- When comparing different `--mem-fraction-static` values and their impact on KV pool capacity
- When planning deployment capacity: how many concurrent requests can a given GPU configuration support
- When investigating OOM issues: identifying which memory category is consuming the most
- When evaluating whether fp8 KV cache or EP can improve concurrency

## Key Concepts

### mem-fraction-static

Controls what fraction of **available GPU memory after weight loading** is reserved for the KV cache pool. Higher values give more KV capacity but less headroom for CUDA graph and other runtime buffers.

- `0.88` (default): aggressive — 88% of post-weight memory goes to KV pool
- `0.60`: conservative — more free memory left for runtime, but significantly less KV capacity

### KV Head Replication

When `num_key_value_heads < tp_size`, KV cache is **replicated** across all TP ranks rather than split. For example, models with `kv_heads=1, tp=8` means each of the 8 cards stores a full copy of the KV cache — 8x the per-card KV memory compared to a split scenario.

### SWA (Sliding Window Attention) Compression

Models like DeepSeek-V4 use CSA (Compressed Sliding Attention) and HCA (Hierarchical Context Attention) with sliding windows. This dramatically reduces per-token KV cache bytes compared to the theoretical full-attention calculation. The `bytes_per_full_token` reported in the log already accounts for this compression.

## Output Contract

Return:

1. **Serving configuration**: model, GPU, TP/PP/EP, mem-fraction-static, kv-cache-dtype
2. **Memory breakdown table**: category / GiB / MiB / percentage / derivation source
3. **Per-rank nvidia-smi comparison**: used and free memory per TP rank
4. **KV pool detail**: pool size, bytes_per_full_token, KV dtype, replication factor, theoretical per-token KV calculation (when config.json provided)
5. **Concurrency estimate table**: request token length / token-limit / request-limit / max concurrent
6. **Tuning suggestions**: actionable recommendations based on free memory and configuration

## Known Limitations

| Limitation | Detail | Workaround |
|---|---|---|
| SGLang-specific patterns | Currently only SGLang log patterns are fully supported | vLLM patterns to be added as encountered |
| SWA compression models | Per-token KV bytes cannot be independently calculated from model config for CSA/HCA attention — the framework's internal SWA window parameters are needed | Use `bytes_per_full_token` from the log directly |
| DeepGEMM JIT memory | The analyzer categorizes DeepGEMM JIT compilation memory as "other" because it is not explicitly reported in the log | Compare with nvidia-smi total for accurate accounting |
| PP (Pipeline Parallelism) | Memory decomposition is per-rank; PP configurations may have uneven memory across stages | Specify `--target-rank` for each PP stage |
| MoE expert buffer | Some frameworks allocate additional buffers for expert routing that are not separately reported | Included in "model weights" or "other" depending on when allocated |

## References

- `references/log-patterns.md`: log line patterns and their semantics for memory analysis.
- `scripts/capacity_analyzer.py`: the core analysis script.
- `../model-compute-simulation/references/gpu-specs.json`: GPU HBM specifications (shared reference).
