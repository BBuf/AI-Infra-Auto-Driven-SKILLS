# AKO4ALL Harness Templates

These four files are scaffolding for one optimization loop. They are not
auto-discovered by any tool — copy them into a fresh `AKO4ALL/<your-task>/`
directory and edit them in-place.

| File | Destination in AKO4ALL | Purpose |
| --- | --- | --- |
| `ITERATIONS.md` | `<task>/ITERATIONS.md` | Hypothesis / change / correctness / bench / decision per iteration |
| `kernel_notes.md` | `<task>/context/<kernel>_notes.md` | Kernel contract: shape, dtype, layout, alignment, baseline bottleneck, failed attempts |
| `bench_kernel.py` | `<task>/bench/bench_<kernel>.py` | Microbench skeleton for Triton / CuTe DSL Python kernels |
| `bench_kernel.cu` | `<task>/bench/bench_<kernel>.cu` | Microbench skeleton for CUDA C++ / CUTLASS-CuTe C++ kernels |

Recommended on-disk layout for a single kernel optimization task:

```text
AKO4ALL/<task>/
├── ITERATIONS.md
├── input/
│   ├── reference.py            # trusted reference (or .cu)
│   └── <kernel>.py             # current target-repo baseline copied in
├── solution/
│   └── <kernel>.py             # candidate(s) under iteration
├── bench/
│   ├── bench_<kernel>.py       # adapted from bench_kernel.py
│   └── (or .cu for CUDA)       # adapted from bench_kernel.cu
└── context/
    └── <kernel>_notes.md       # adapted from kernel_notes.md
```

Once you have these four files filled in, the AKO loop is just:

1. Run `bench --target baseline` once, paste the table into `ITERATIONS.md`
   under "Baseline Snapshot".
2. For each iteration:
   - one-line hypothesis in `ITERATIONS.md`
   - apply one change in `solution/<kernel>.py` (or `.cu`)
   - correctness first (`bench --target candidate` fails fast on mismatch)
   - paste the new row into `ITERATIONS.md`
   - decide ✅ keep / 🔄 revert / ⚠️ flag.
3. After 3 no-improvement iterations or any other stop condition, rerun `ncu`
   on the current best, reread the iteration log, and switch direction.
