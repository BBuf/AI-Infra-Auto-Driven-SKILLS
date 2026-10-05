# Serving Log Patterns for Memory Analysis

This document catalogs the log line patterns used by `capacity_analyzer.py` to extract memory data from LLM serving framework startup logs.

## Legacy SGLang Patterns

These examples remain parser fallbacks; use the current formats below for main.

### 1. Server Arguments

```
[timestamp] server_args=ServerArgs(model_path='...', tp_size=8, pp_size=1, mem_fraction_static=0.88, kv_cache_dtype='fp8_e4m3', ...)
```

Extracts: `model_path`, `tp_size`, `pp_size`, `dp_size`, `ep_size`, `mem_fraction_static`, `kv_cache_dtype`, `cuda_graph_max_bs`, `disable_radix_cache`, `page_size`, `swa_full_tokens_ratio`

### 2. Load Weight Begin

```
[2026-05-15 08:39:36 TP0] Load weight begin. avail mem=93.61 GB
```

Extracts: rank, avail_gb (free GPU memory before weight loading)

This is the baseline for all subsequent memory calculations:
- `framework_overhead = GPU_HBM - avail_before_weight`

### 3. Legacy Memory Profiling (not emitted by current main)

```
[2026-05-15 09:09:53 TP0] Memory profiling: available_gpu_memory=57.01 GB, total_gpu_memory=93.58 GB, mem_fraction_static=0.60, rest_memory=19.58 GB
```

Extracts: rank, available_gpu_memory, total_gpu_memory, mem_fraction_static, rest_memory

Key semantics:
- `total_gpu_memory`: historical reservation baseline, not post-weight free memory
- `available_gpu_memory`: currently free memory before allocating the KV pool
- For this example, `57.01 - 93.58 * (1 - 0.60) ≈ 19.58 GiB`.
  Current sizing also accounts for model-specific reservations and post-capture
  resizing; see [source contracts](../../../docs/upstream-source-contracts.md).
- `rest_memory`: actual memory reserved for KV pool (after subtracting framework buffers from `available_gpu_memory`)

### 4. SW KV Memory Calculation (SWA models)

```
[2026-05-15 09:09:53 TP0] DSv4 memory calculation: bytes_per_full_token=15955.85, available_bytes=19.58 GB, full_token=1317632
```

Extracts: rank, bytes_per_full_token, available_bytes_gb, full_token

Key semantics:
- `bytes_per_full_token`: actual KV cache bytes per token including SWA (Sliding Window Attention) compression
- `available_bytes`: same as `rest_memory` above (the KV pool budget)
- `full_token`: `available_bytes / bytes_per_full_token` = max number of full tokens in KV pool

### 5. Memory Pool End

```
[2026-05-15 09:09:54 TP0] Memory pool end. avail mem=36.31 GB
```

Extracts: rank, avail_gb (free memory after KV pool allocation)

This marks the point where KV pool has been allocated. The difference from `load_weight_begin` gives:
- `weight + kv_pool = avail_before_weight - avail_after_pool`

### 6. CUDA Graph Capture End

```
[2026-05-15 08:40:32 TP0] Capture cuda graph end. Time elapsed: 54.61 s. mem usage=1.93 GB. avail mem=8.16 GB.
```

Extracts: rank, elapsed_s, mem_usage_gb, avail_gb

Key semantics:
- `mem_usage_gb`: CUDA graph memory consumption (graph buffers for all batch sizes)
- `avail_gb`: remaining free memory after CUDA graph capture

### 7. Final Token Capacity

```
[2026-05-15 08:40:32 TP0] max_total_num_tokens=3080960, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=256, context_len=1048576, available_gpu_mem=8.16 GB
```

Extracts: max_total_num_tokens, chunked_prefill_size, max_prefill_tokens, max_running_requests, context_len, available_gpu_mem_gb

Key semantics:
- `max_total_num_tokens`: total token capacity across all requests (determines max concurrency)
- `max_running_requests`: scheduler limit on concurrent requests
- Max concurrent requests = `min(max_total_num_tokens / tokens_per_request, max_running_requests)`

## Memory Decomposition Logic

### With Memory Profiling Line (newer sglang)

```
framework_overhead = GPU_HBM - avail_before_weight
model_weights      = avail_before_weight - memory_profiling.available_gpu_memory
kv_pool            = memory_profiling.rest_memory
cuda_graph         = cuda_graph_end.mem_usage
other              = nvidia_smi_used - sum(above)
```

### Without Memory Profiling Line (older sglang)

```
framework_overhead = GPU_HBM - avail_before_weight
weight + kv_pool   = avail_before_weight - avail_after_pool
kv_pool            = sw_kv_calc.available_bytes  (or inferred from other data)
model_weights      = (avail_before_weight - avail_after_pool) - kv_pool
cuda_graph         = cuda_graph_end.mem_usage
other              = nvidia_smi_used - sum(above)
```

## nvidia-smi Output Format

Expected format (from `nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader`):

```
0, 89846 MiB, 7522 MiB
1, 89942 MiB, 7426 MiB
2, 89942 MiB, 7426 MiB
...
```

This provides per-rank memory comparison, useful for identifying uneven distribution.

## vLLM Patterns

These patterns match the current vLLM V1 sources in
`vllm/v1/worker/gpu_worker.py`,
`vllm/v1/worker/gpu_model_runner.py`, and
`vllm/v1/core/kv_cache_utils.py`.

```text
Initial free memory: 79.20 GiB; Requested memory: 0.900000 (util), 72.00 GiB
Model loading took 14.50 GiB memory and 12.000000 seconds
Available KV cache memory: 52.25 GiB
Graph capturing finished in 8 secs, took 1.25 GiB
GPU KV cache size: 1,572,864 tokens
Maximum concurrency for 8,192 tokens per request: 192.00x
```

The analyzer records initial/requested memory, model-loading memory, available
KV-cache memory, CUDA-graph memory, total GPU KV tokens, the model length used
for vLLM's calculation, and its reported theoretical concurrency.

vLLM and SGLang do not expose identical checkpoints. A missing vLLM field is
reported as unknown; the analyzer does not synthesize
`mem_fraction_static`, `cuda_graph_max_bs`, or SGLang memory-pool events.

### vLLM Memory Decomposition

```text
framework_overhead = GPU_HBM - initial_free_memory
model_weights      = model_loading_memory
kv_pool            = available_kv_cache_memory
cuda_graph         = graph_capture_memory
other              = nvidia_smi_used - sum(above)
```

Without `nvidia-smi`, the known fields are summed and the unreported residual
remains unknown rather than being assigned to a fabricated category.

## Current source formats — 2026-10-05

- Resolved config: `server_args={'model_path': '...', 'tp_size': 8, ...}`.
- Weight end: `Load weight end. elapsed=X s, type=<Cls>, avail mem=X GB, mem usage=X GB.`
- DSV4: `DSV4 memory calculation: unified=..., bytes_per_full_token=..., available_bytes=... GB, c128_state_fixed=... GB, c2_state_fixed=... GB, swa_fixed=... GB, swa_ring_fixed=... GB, c4_state_fixed=... GB, full_token=N`. Extract fields by name.
- KV pools: `[label] KV Cache is allocated. dtype: <dt>, #tokens: N, K size: X GB, V size: Y GB`, or `KV size: X GB`. Sum pools once per rank. `KV Cache VA upper bound` is not an allocation.
- Recurrent pools: `Mamba Cache is allocated. ... conv_state size: XGB, ssm_state size: YGB`; speculative variants also log intermediate SSM and physical conv-window buffers. Keep recurrent allocations separate from KV resizing.
- Graph capture: `Capture target|draft decode|verify|prefill CUDA graph end. elapsed=X s, mem usage=X GB, avail mem=X GB.` Sum phase/role captures per rank.
- Opt-in `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=1`: final `Post-capture KV sizing: KV cache allocated. ... KV size: X GB, avail mem=X GB` replaces the target KV estimate. A current target logs a VA upper bound before capture; regular allocation lines alongside it can belong to independent draft pools and must be retained. Draft workers do not run post-capture sizing. `Memory pool end` precedes final resizing.
- Final limits accept `chunked_prefill_size=-1` and `available_cpu_mem` on CPU.
- Prefix tags are optional, e.g. `[time]` on one GPU or `[time DP0 PP0 ATTN_CP0 MOE_DP0 TP0 EP0]`. Extract TP independently of tag order.

vLLM INFO exposes `Free memory on device (F/T GiB) on startup. Desired GPU memory utilization is (U, R GiB). Actual usage is C GiB for consumed memory (weights + non-torch), P GiB for peak activation, and G GiB for CUDAGraph memory. ... Current kv cache memory in use is K GiB.` Prefer T or measured `nvidia-smi` totals over marketing GB.
The explicit `kv_cache_memory_bytes` path instead logs reserved KV bytes and skips profiling. The old initial-free line requires DEBUG. `<device> KV cache size` is device-dependent. Its maximum-concurrency figure is capacity at max length, not the scheduler cap: only use logged `non-default args: {'max_num_seqs': N}` as a request limit; otherwise report it unknown.

MLA stores one latent plus RoPE key: `layers * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes` per GPU, with replicated latent. FP4 payload is 0.5 bytes/element, excluding scales, padding and packing overhead. DSV4 compressed pools require logged sizing, which includes physical page padding (#41091).

Evidence: [pool configurator](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/srt/model_executor/pool_configurator.py),
[memory pools](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/srt/mem_cache/memory_pool.py),
[graph setup](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py),
[vLLM memory logs](https://github.com/vllm-project/vllm/blob/0c16eee3f1ff777298cc894c3eeb85f3880c6d6a/vllm/v1/worker/gpu_worker.py).
