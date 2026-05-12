# KernelPilot Kernel Goal Prompt

Copy the prompt below into Codex. In the common case, fill only the operator
and target speedup. Leave the optional fields unset when you want Codex to
discover the baseline, tests, benchmark, and hardware configuration.

```text
Use the KernelPilot workflow in this repository to continuously optimize one
kernel.

- Operator: <kernel operator name or entry point, for example rmsnorm_hf / timestep_embedding / group_norm_silu>
- Target speedup: <for example 1.10x; every configured case must be no slower than baseline and geomean speedup must reach this target>
- Target hardware: <optional; leave unset to let Codex use the available GPU validation environment>
- Target repository: <optional target kernel project worktree or branch; leave unset to let Codex create or use an isolated clean worktree>
- Runtime environment: <optional; user-managed shell, SSH host, CI runner, or other execution environment>
- KernelPilot root: <path to the KernelPilot directory in the current workspace, or leave unset if already running from it>

Do not stop at a plan. Keep iterating until the target is met or a concrete
blocker is proven. Requirements:

1. Locate the operator's Python/CUDA/Triton entry points, tests, benchmarks,
   shape and dtype distributions, and current baseline in the target kernel
   project.
2. Create or update a KernelPilot task directory for this operator. Record the
   semantic spec, forbidden files, correctness gates, benchmark cases, baseline,
   and lineage.
3. Follow the KernelPilot single-lineage loop: one hypothesis at a time, a
   from-scratch candidate, correctness first, then a full benchmark.
4. Use references only to form hypotheses and validation methods. Do not copy
   external kernel code. Approved references include:
   - local GPU-kernel reference docs configured by the user;
   - FlashInfer;
   - FlashAttention;
   - Tencent/hpc-ops;
   - DeepGEMM.
   The target project is used for wrappers, tests, and benchmarks. Do not use
   existing target kernels as external reference implementations.
5. If a direction fails, record the reason, data, and next decision in lineage
   notes. Do not repeat the same direction without new evidence.
6. If three consecutive iterations do not improve the best candidate, reread
   lineage, benchmark distributions, and profiler evidence, then switch
   direction.
7. Finish only when all gates pass:
   - correctness tests pass;
   - the benchmark covers every configured case;
   - every case has speedup >= 1.0x;
   - geomean speedup >= target speedup;
   - the result is validated on the selected hardware, when hardware is
     specified or available;
   - KernelPilot run artifacts, lineage, and final comparison are updated.
8. Final response must include:
   - changed or added file paths;
   - baseline vs candidate table;
   - acceleration or failure source for each major candidate version;
   - correctness and performance commands that were run;
   - whether the result is integrated into the target project or remains a standalone
     harness.
```

Optional fields:

```text
- Baseline: <source change link, commit, benchmark JSON, or "measure current main">
- Shape/dtype cases: <target workload distribution, if known>
- Forbidden files: <stricter file list, if needed>
- Acceptance tests: <pytest, benchmark, or server smoke commands>
```
