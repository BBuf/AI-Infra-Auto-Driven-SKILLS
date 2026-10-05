---
name: llm-serving-capacity-planner
description: "Parse SGLang/vLLM startup logs to explain GPU memory use and request capacity. Use for KV cache budget, mem-fraction-static comparisons, OOM triage, and max-concurrency estimates."
---

# LLM Serving Capacity Planner

## Overview

Use this when a serving log has enough memory lines to explain where GPU HBM
went. The analyzer reads SGLang/vLLM startup logs, extracts weight load, KV
pool, CUDA graph, framework overhead, and token-capacity lines, then estimates
concurrent requests for common token lengths.

For DeepSeek-V4.1, read the
[KV layout and page-allocation contract](../llm-torch-profiler-analysis/references/fuse-overlap-catalog.md#paged-kv-storage-contract)
before using 528 B/288 B in an estimate. These are per-stored-token payload and
scale sizes, not allocation after page padding or total model KV bytes. Verify
actual pool layout, compression ratios and sharing from the pinned source and logs.

## Inputs

Before running analysis, collect or verify these inputs:

| Item | Why it matters | How to obtain | Default if user skips |
|---|---|---|---|
| Log file path | Primary input; all memory data comes from here | Ask user for the serving startup log | — (required) |
| GPU type | Determines total HBM for decomposition validation | Ask user or infer from log | Auto-detected from log if possible |
| nvidia-smi output | Provides per-rank actual memory for cross-validation | Capture with `nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader > smi.txt` | — (optional, but recommended) |
| Model config.json | Enables theoretical KV cache byte calculation and replication factor analysis | Ask user for the model's config.json path | — (optional, log data used instead) |
| Request token length | Determines concurrency estimate denominator | Ask user | 4096, 6144, 8192 |

## Workflow

### Step 1: Collect the serving log

The user should provide the startup log from an SGLang or vLLM serving instance. Key log lines that the analyzer needs:

- `Load weight begin. avail mem=XX GB`
- `server_args={'model_path': '...', 'tp_size': 8, ...}` (resolved parameters)
- `Load weight end. ... avail mem=X GB, mem usage=X GB.` (weight footprint)
- `DSV4 memory calculation: ... bytes_per_full_token=X, available_bytes=X GB, ... full_token=N`
- `[label] KV Cache is allocated. dtype: ..., #tokens: N, K size: X GB, V size: Y GB` (or `KV size`)
- `Mamba Cache is allocated. ... conv_state size: X GB, ssm_state size: Y GB`
- `Memory pool end. avail mem=XX GB`
- `Capture target|draft decode|verify|prefill CUDA graph end. elapsed=X s, mem usage=X GB, avail mem=X GB.`
- `Post-capture KV sizing: KV cache allocated. ... KV size: X GB, avail mem=X GB` (opt-in)
- `max_total_num_tokens=XX, ... chunked_prefill_size=-1|N, ... available_gpu_mem=XX GB`

Legacy `ServerArgs(...)`, `Memory profiling:` and `Capture cuda graph end.`
remain accepted for older/private branch logs; they are not current main emitters.

Current vLLM V1 logs instead expose:

- `Initial free memory: ...; Requested memory: ...` (DEBUG only)
- `Model loading took ... GiB`
- `Available KV cache memory: ... GiB`
- `Graph capturing finished ... took ... GiB`
- `<device> KV cache size: ... tokens, Maximum concurrency for ... tokens per request: ...x`
- `Free memory on device (F/T GiB) ... Current kv cache memory in use is K GiB` (INFO)
- `CUDA graph pool memory: A GiB (actual), E GiB (estimated), ...`
- `Initial free memory X GiB, reserved Y GiB memory for KV Cache ...` (explicit KV bytes)
- `Maximum concurrency for ... tokens per request: ...x`

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

The analyzer prints:

1. **Memory breakdown table**: each category (weights, KV pool, CUDA graph, framework, other) with GiB, MiB, percentage, and derivation
2. **Per-rank comparison**: nvidia-smi data across all TP ranks
3. **KV pool detail**: pool configuration, KV dtype, replication factor, per-token byte calculation
4. **Concurrency estimate**: max concurrent requests for different token lengths
5. **Tuning notes**: configuration changes that may increase capacity

For vLLM, the report also includes a `vLLM Startup Evidence` section. Values
that are not present in the log stay unknown; do not translate them into
SGLang-only fields.

## When To Use It

- After launching an LLM serving instance, to understand how GPU memory is distributed
- When comparing different `--mem-fraction-static` values and their impact on KV pool capacity
- When planning deployment capacity: how many concurrent requests can a given GPU configuration support
- When investigating OOM issues: identifying which memory category is consuming the most
- When evaluating whether fp8 KV cache or EP can improve concurrency

## Key Concepts

### mem-fraction-static

The inspected SGLang source reserves runtime headroom from **pre-model-load
free memory**: `headroom = pre_model_load_memory * (1 - mem_fraction_static)`.
The KV budget starts from currently free memory minus this headroom and other
model-specific reservations. It is not `post_weight_free * fraction`.
Post-capture sizing may update the pool again. Use final logged capacity and
rank-local evidence; see [source contracts](../../docs/upstream-source-contracts.md).

The default is auto-resolved from the hardware and configuration. Read the
resolved value from `server_args={...}`. A lower value increases runtime
headroom at the cost of KV capacity. Do not infer 0.88 when it is absent.

### KV Head Replication

For standard GQA/MHA, each rank stores `max(1, num_key_value_heads // tp_size)` KV heads. When TP exceeds the head count, each head is replicated across a subgroup of ranks, not the whole head set on every rank. For `kv_heads=4, tp=8`, each rank stores one head; for `kv_heads=1, tp=8`, all eight ranks store that single head. MLA stores a replicated latent plus RoPE key instead. The theoretical calculation assumes all configured layers reside on the rank; use rank-local layer counts for pipeline parallelism.

### SWA (Sliding Window Attention) Compression

Models like DeepSeek-V4 use CSA (Compressed Sliding Attention) and HCA
(Hierarchical Context Attention) with sliding windows. This reduces per-token
KV cache bytes compared to the theoretical full-attention calculation. The
`bytes_per_full_token` reported in the log already accounts for this
compression.

## Reporting Checklist

Include:

1. **Serving configuration**: model, GPU, TP/PP/EP, mem-fraction-static, kv-cache-dtype
2. **Memory breakdown table**: category / GiB / MiB / percentage / derivation source
3. **Per-rank nvidia-smi comparison**: used and free memory per TP rank
4. **KV pool detail**: pool size, bytes_per_full_token, KV dtype, replication factor, theoretical per-token KV calculation (when config.json provided)
5. **Concurrency estimate table**: request token length / token-limit / request-limit / max concurrent
6. **Tuning notes** based on free memory and configuration

## 2026-08-23 Runtime Notes

SGLang `v0.5.18` consolidates compiled-kernel caches under `SGLANG_CACHE_DIR`.
The first launch after an upgrade can look like extra "other" memory and a
longer startup because Triton / FlashInfer / DeepGEMM / Inductor recompile
once. Overlapped checkpoint staging (`--startup-weight-load-mode overlap`)
also changes the startup memory curve: weight pages and CUDA-graph capture
can overlap, so do not treat a shorter load-weight interval as a smaller
weight footprint.

## Known Limitations

| Limitation | Detail | Workaround |
|---|---|---|
| Framework-specific checkpoints | SGLang and vLLM expose different memory checkpoints, so some fields are not comparable one-to-one | Report unknown fields and compare only shared, explicit evidence |
| SWA compression models | Per-token KV bytes cannot be independently calculated from model config for CSA/HCA attention — the framework's internal SWA window parameters are needed | Use `bytes_per_full_token` from the log directly |
| DeepGEMM JIT memory | The analyzer categorizes DeepGEMM JIT compilation memory as "other" because it is not explicitly reported in the log | Compare with nvidia-smi total for accurate accounting |
| PP (Pipeline Parallelism) | Memory decomposition is per-rank; PP configurations may have uneven memory across stages | Specify `--target-rank` for each PP stage |
| MoE expert buffer | Some frameworks allocate additional buffers for expert routing that are not separately reported | Included in "model weights" or "other" depending on when allocated |

## References

- `references/log-patterns.md`: log line patterns and their semantics for memory analysis.
- `references/gpu-specs.json`: GPU HBM specifications for `h20`, `h100`, `h200`, and `b200` aliases.
- `scripts/capacity_analyzer.py`: the core analysis script.

## Memory capacity sources

The bundled hardware table records vendor GB maxima, not usable GiB. Prefer
`nvidia-smi` used+free or vLLM's logged total. B200 retains the DGX 180GB SKU;
other Blackwell SKUs may differ. DGX Spark's 128GB is unified system memory,
not dedicated HBM. L20 is the official 48GB SKU; do not infer a capacity for
`l20n` from the name. GB200 records the physical 192GB package maximum; its deployed SKU and
custom SKUs require measured totals. Official
source URLs are stored with each newly added hardware entry.
