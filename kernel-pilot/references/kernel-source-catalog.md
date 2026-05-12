# Kernel Source Catalog

This is the KernelPilot knowledge base `K` for GPU-kernel work. It tells the agent what
to read before proposing an iteration, and which sources are only references
rather than candidate-code inputs.

The machine-readable version is `kernel-source-catalog.json`.

## Local References

- User-configured GPU-kernel reference docs
  - Read `ako4all-kernel-loop.md` for the iteration discipline.
  - Read `cuda-cpp-kernel-reference.md`, `triton-kernel-reference.md`,
    `cute-dsl-kernel-reference.md`, or `cutlass-cpp-kernel-reference.md`
    according to the implementation lane.
  - Read `profiling-debugging-reference.md` before making a performance claim.
  - Read `nvidia-architecture-reference.md` and the matching SM guide for
    architecture-specific work.

- User-configured target kernel project worktree
  - Clean `origin/main` worktree for integration and validation.
  - Useful paths: `src/kernels`,
    `tests/kernels`, `benchmarks`,
    `include/kernel_project`, and `kernel-library`.

## External Repositories

| Repo | Verified HEAD | Read first |
| --- | --- | --- |
| `flashinfer-ai/flashinfer` | `9f4d16174e81dc818ca65f9eb930ec5fd8a9b395` | `csrc`, `csrc/fmha_v2/fmha`, `csrc/fmha_v2/fmha/hopper`, `csrc/fmha_v2/fmha/warpspec`, `benchmarks/bench_blackwell_attention.py` |
| `Dao-AILab/flash-attention` | `ab66326aaa4fe3529fbc00f3156f3a762dd3141b` | `csrc/flash_attn/src`, `flash_attn/cute`, `benchmarks/benchmark_attn.py`, `benchmarks/bench_sm90.py` |
| `Tencent/hpc-ops` | `8d300e1be9e5e2089e1da3c134f1e237b9063e21` | `hpc/attention.py`, `hpc/group_gemm.py`, `hpc/normalization.py`, `hpc/rope.py`, `src/attention`, `src/group_gemm`, `src/normalization`, `src/rope`, `tests/` |
| `deepseek-ai/DeepGEMM` | `714dd1a4a980f7937a74343d19a8eba4fe321480` | `csrc/apis`, `csrc/jit`, `csrc/jit_kernels/heuristics`, `csrc/jit_kernels/impls`, `deep_gemm/include/deep_gemm`, `deep_gemm/testing`, `tests/` |

Target-project source is available only for integration, tests, and benchmarks
through the local clean worktree above. It is not an external implementation
reference.

## Usage Rules

1. Read references to identify hypotheses, contracts, and measurement style.
2. Do not copy external kernel code into a candidate.
3. For target-project task files, `forbidden_files` cannot seed a from-scratch
   candidate. They may be read later for integration or old-vs-new comparison.
4. Every source-derived idea must be logged in lineage notes with the source
   path and the exact hypothesis being tested.
