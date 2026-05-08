# Profiling and Debugging Reference

Based on the profiling and debugging material in the cache-dit CUDA/CuTe/CUTLASS kernel skills.

Read this file before using a profiler result to justify a kernel change, before claiming a speedup, or when debugging correctness failures.

## Tool Order

Use tools in this order unless the failure mode demands otherwise:

1. focused correctness reproducer
2. compute-sanitizer or cuda-gdb for memory/race/sync bugs
3. `nsys` for timeline, launch gaps, CPU/GPU interaction, copies, and overlap
4. `ncu` for per-kernel root cause
5. PTX/SASS inspection when compiler codegen or instruction selection is suspect

## compute-sanitizer

Use:

```bash
compute-sanitizer --tool memcheck ./program
compute-sanitizer --tool racecheck ./program
compute-sanitizer --tool initcheck ./program
compute-sanitizer --tool synccheck ./program
```

Interpretation hints:

- out-of-bounds shared-memory errors can mean insufficient dynamic shared memory allocation
- errors only on some lanes or thread ranges often indicate warp-boundary assumptions
- shape-specific failures in async kernels often indicate barrier, predicate, or stage reuse bugs

## cuda-gdb and Binary Inspection

Use cuda-gdb in batch mode for crashes:

```bash
cuda-gdb -batch -ex "run" -ex "bt" -ex "info cuda threads" ./program
```

Use `cuobjdump` for resource usage and code inspection:

```bash
cuobjdump -res-usage ./program
cuobjdump -ptx ./program
cuobjdump -sass ./program
```

Compile with `-lineinfo` for profiler/source correlation.

## Nsight Systems

Use `nsys` to answer "where is time spent?"

```bash
nsys profile --trace=cuda,nvtx,osrt -o report ./program
nsys stats report.nsys-rep --report cuda_gpu_kern_sum
nsys stats report.nsys-rep --report cuda_api_sum
nsys stats report.nsys-rep --report cuda_gpu_mem_time_sum
nsys stats report.nsys-rep --report nvtx_sum
```

Look for:

- many kernels under 10 us
- gaps between kernels
- `cudaDeviceSynchronize` or allocation in steady state
- memory copies that should be removed or overlapped
- missing stream overlap

## Nsight Compute

Use `ncu` to answer "why is this kernel slow?"

```bash
ncu --set basic --kernel-name "kernel" ./program
ncu --section SpeedOfLight --kernel-name "kernel" ./program
ncu --section Occupancy --section LaunchStatistics --kernel-name "kernel" ./program
ncu --section MemoryWorkloadAnalysis --kernel-name "kernel" ./program
```

Classify:

- SM high, memory low: compute-bound
- memory high, SM low: memory-bound
- both low: latency, occupancy, scheduling, or synchronization issue

Useful metric families:

- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__warps_active.avg.pct_of_peak_sustained_elapsed`
- `launch__occupancy_limit_registers`
- `launch__occupancy_limit_shared_mem`
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum`
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`
- `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`

## NVTX

Use NVTX when kernel-level granularity is not enough:

```cpp
nvtxRangePush("region");
// work
nvtxRangePop();
```

Profile with:

```bash
nsys profile --trace=cuda,nvtx -o report ./program
nsys stats report.nsys-rep --report nvtx_sum
```

Keep ranges coarse, stable, and nested by phase.

## Common Performance Traps

- bank conflicts from transpose-like shared-memory access
- poor coalescing from strided global loads
- too many tiny kernels and launch gaps
- optimization tuned for a scale different from production
- async copy overhead without enough compute to overlap
- TMA, WGMMA, tcgen05, or cluster features used where setup cost dominates
- failing to record negative results, leading to repeated failed attempts

The loop is: profile, form one hypothesis, change one thing, verify, record.
