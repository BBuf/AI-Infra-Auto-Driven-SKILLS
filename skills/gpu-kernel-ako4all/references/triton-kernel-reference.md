# Triton Kernel Reference

Based on https://github.com/anthony-maio/triton-skills.

Read this file before writing or changing Triton kernels under `gpu-kernel-ako4all`.

## Core Patterns

- Use `@triton.jit`.
- Use `tl.program_id(axis)` for block IDs.
- Build offsets with `tl.arange(0, BLOCK_SIZE)`.
- Declare tile sizes as `tl.constexpr`.
- Prefer power-of-two block sizes for `tl.arange`.
- Mask every boundary `tl.load` and `tl.store`.
- Accumulate FP16/BF16 inputs in `tl.float32`.
- Cast to output dtype only at store time.
- Pass strides explicitly from the Python launcher instead of assuming contiguous layout.
- Provide a PyTorch fallback or reference when Triton is optional.

## Launcher Rules

- Use dynamic grids:

```python
grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
```

- For attention-like kernels, use a 2D grid:

```python
grid = (triton.cdiv(Q_LEN, BLOCK_M), B * H)
```

- For GEMM-like kernels, use one program per output tile and map linear tile IDs with `//` and `%`.
- For inference paths where autotune warmup is too expensive, write a Python launcher that selects `BLOCK_M`, `BLOCK_N`, `num_warps`, and `num_stages` from shape, dtype, and target GPU.

## Common Optimization Families

### Tiled GEMM

- Use `tl.dot(a, b)` and FP32 accumulators.
- Use grouped tile ordering for L2 locality.
- Include at least one small tile config for small shapes.
- Use `tl.max_contiguous` and `tl.multiple_of` when offsets are known aligned.

### Flash Attention

- Use online softmax with `(m, l, acc)` in registers.
- Load K/V tiles once per loop and reuse them across streams when possible.
- For causal attention with KV cache, use lower-right causal masking:

```python
prefix_len = KV_LEN - Q_LEN
causal_mask = (offs_m[:, None] + prefix_len) >= offs_kv[None, :]
```

- Guard all-masked rows by adding a small epsilon to the softmax denominator.
- Apply `sm_scale` after `tl.dot`; avoid pre-scaling Q when that changes dtype behavior.

### Fused Epilogues

- Fuse normalization, gating, bias, residual, activation, or dropout immediately before `tl.store`.
- Use `tl.constexpr` feature flags so disabled paths compile away.
- Load small vectors such as bias, gate, or norm weights outside K loops.
- Monitor register pressure for multi-stream attention or fused epilogues.

### Normalization

- Map one program per row or group.
- Use FP32 for reductions.
- For RMSNorm, compute `rstd = tl.math.rsqrt(mean(x*x) + eps)`.
- Use two-stage reductions for backward `dgamma`/`dbeta` when atomics would contend.

### Persistent and Warp-Specialized Matmul

- Use persistent scheduling when fewer programs than output tiles can improve utilization.
- Launch at most `NUM_SMS` persistent programs for tile loops.
- Query `torch.cuda.get_device_properties(...).multi_processor_count`; do not hardcode SM count.
- On Hopper+ paths, consider TMA descriptors and `tl.range(..., flatten=True)` or warp specialization.

### Quantized or Block-Scaled GEMM

- Prefer hardware scaled dot paths on supported hardware.
- Keep scale tensor layout explicit.
- For fallback paths, unpack or dequantize tiles to FP16/FP32 and accumulate safely.
- Test tails, signed 4-bit behavior, symmetric/asymmetric quantization, and scale layout.

### Sequential Stateful Blocks

- Use one block per independent sequence when iteration `t` depends on mutations from `t-1`.
- Load mutable state into registers before the loop.
- Mutate register state with `tl.where`; Triton tensor index assignment is not valid.
- Write final state to HBM only after the sequential loop.

## Profiling Questions

Use `ncu` to classify:

- memory-bound: DRAM high, compute low
- compute-bound: tensor/SM utilization high, DRAM lower
- underutilized: both low, high stall metrics or occupancy limits

Then choose the change that matches the category. Do not sweep tile sizes blindly.

## Pitfalls

- Missing masks silently corrupt memory.
- Assuming contiguous tensors breaks after transpose, permute, or GQA expansion.
- Python `divmod` is not supported in Triton JIT; use `//` and `%`.
- Large fused kernels can lose from register pressure even when memory traffic drops.
- Autotune can be unacceptable in hot real-time inference launch paths.
