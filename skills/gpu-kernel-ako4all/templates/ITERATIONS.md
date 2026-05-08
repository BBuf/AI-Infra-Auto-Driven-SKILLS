# Iterations Log — `<kernel-name>`

> Copy this file into `AKO4ALL/<your-task>/ITERATIONS.md` at the start of an
> AKO loop. Append one entry per iteration. Keep failed attempts; they are the
> evidence trail that decides when to switch direction.

## Task Header

| Field | Value |
| --- | --- |
| Kernel | `<entry-point name>` |
| Implementation lane | Triton / CUDA C++ / CUTLASS-CuTe C++ / CuTe DSL |
| Target GPU(s) | e.g. H100 (sm90), B200 (sm100), RTX 5090 (sm120) |
| Shape family / dtype / layout | e.g. `(B, H, M=4096, K=1024) bf16, NT, contiguous` |
| Hot shape (first to optimize) | `(B=1, H=32, M=4096, N=4096, K=128)` |
| Bottleneck class (compute / memory / launch / sync / layout-conv) | filled after baseline ncu/nsys |
| Reference (trusted) | PyTorch eager / previous operator / paper / etc. |
| Best baseline runtime (us / TFLOP/s) | from baseline microbench |

## Baseline Snapshot

| Metric | Value | Source |
| --- | --- | --- |
| median (us) | | bench/bench_<kernel>.py |
| p95 (us) | | bench/bench_<kernel>.py |
| sm__throughput | | ncu basic |
| dram__throughput | | ncu basic |
| occupancy (warps) | | ncu Occupancy |
| stall reasons (top-3) | | ncu WarpState |
| launch gaps observed? | yes / no | nsys |

## Iteration Entries

Use one block per iteration. **One change per iteration.**

### Iter 0 — baseline reproduction (no edits)

- Hypothesis: AKO baseline matches the reference (parity check).
- Command: `python bench/bench_<kernel>.py --shape <hot> --iters 100 --warmup 20`
- Result: median = ___ us, max-abs-err vs reference = ___
- Decision: ✅ accept as baseline / ❌ harness still wrong, fix before iter 1.

### Iter N — `<short hypothesis>`

- Hypothesis: e.g. "increase BLOCK_K from 32 to 64; expect DRAM throughput up, register pressure within budget".
- Change (single, narrow): file + diff summary, e.g. `solution/<kernel>.py: BLOCK_K = 64`.
- Correctness check first: `pytest test_<kernel>.py -k <hot> -x` → ✅ / ❌
- Microbench: median = ___ us (Δ vs baseline = ___ %), p95 = ___
- Profiler delta:
  - ncu: sm__throughput ___ → ___, dram__throughput ___ → ___, occupancy ___ → ___
  - nsys: launch gap ___ → ___, kernel count ___ → ___
- Decision: ✅ keep / 🔄 revert and try a different idea / ⚠️ keep but flagged (regression risk).
- Notes: any surprising observation, side effects, shape-specific behavior.

### Stop Conditions Reminder

Switch direction (or escalate to a different lane) when **any** of these holds:

- [ ] 3 consecutive iterations did not beat the best runtime
- [ ] correctness becomes fragile or shape-specific
- [ ] the win depends on unrepresentative shapes (only the hottest shape)
- [ ] the microbench improves but the real operator / model path does not

## Final Snapshot

| Metric | Baseline | Final | Δ |
| --- | --- | --- | --- |
| median (us) | | | |
| p95 (us) | | | |
| sm__throughput | | | |
| dram__throughput | | | |
| occupancy | | | |

Root-cause one-liner: <why did it get faster — say it in one sentence>.

Risk register (untested architectures / layouts / tail shapes / dtype variants):
- ___
- ___
