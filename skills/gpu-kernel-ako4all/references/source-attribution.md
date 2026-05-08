# Source Attribution

This skill is a derivative synthesis and is not original work by this repository.
The source skill materials were copied or integrated into this skill so it is self-contained and does not need to read external skill repositories.

## Upstream Sources

1. **SGLang diffusion AKO4ALL skill** (the AKO4ALL outer-loop framework)
   - Repository path: <https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-ako4all-kernel>
   - Local source path used for the copy: `/Users/bbuf/工作目录/Common/sglang/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-ako4all-kernel`
   - Integrated into: `SKILL.md`, `references/ako4all-kernel-loop.md`, `templates/ITERATIONS.md`, `templates/kernel_notes.md`, `templates/bench_kernel.py`, `templates/bench_kernel.cu`, and `scripts/ensure_ako4all_clean.sh`.
   - Used for: AKO4ALL repo hygiene, AKO harness structure, iteration loop discipline, validation gates, PR artifact expectations.
   - Note: the diffusion-specific upstream skill content is **not retained as a sub-skill** because this skill itself owns the generalized outer workflow. For diffusion-specific validation (denoise / scheduler / LoRA), pair this skill with the original `sglang-diffusion-ako4all-kernel` skill at hand-off time.

2. **Triton skills by Anthony Maio**
   - Repository: <https://github.com/anthony-maio/triton-skills>
   - Copied into: `references/triton/`
   - The original `SKILL.md` is preserved as `references/triton/triton-overview.md` (frontmatter stripped). The 9 specialized topic files are preserved as `references/triton/triton-*.md` (frontmatter also stripped, replaced by HTML-comment provenance lines).
   - Used for: Triton core kernel patterns, launcher tiling, FlashAttention, fused epilogues, fused normalization, persistent matmul, quantized GEMM, memory-efficient patterns, and sequential stateful kernels.

3. **cache-dit CUDA C++ kernel skill**
   - Repository path: <https://github.com/vipshop/cache-dit/tree/main/.copilot/skills/cuda-cpp-kernel>
   - Local source path used for the copy: `/Users/bbuf/工作目录/Common/cache-dit/.copilot/skills/cuda-cpp-kernel`
   - Copied into: `references/cuda-cpp/`
   - The original `SKILL.md` is preserved as `references/cuda-cpp/cuda-cpp-overview.md` with an Adapter Note replacing all `references/...` paths to point at the new `vendored-docs/` and shared top-level locations.
   - The full vendored documentation tree (PTX ISA, CUDA Runtime/Driver, Programming Guide, Best Practices Guide, Nsight Compute, Nsight Systems) is preserved under `references/cuda-cpp/vendored-docs/` (~12 MB). It is opt-in; see `references/cuda-cpp/vendored-docs/README.md` for the trim guide.
   - The architecture-specific guides for sm100 / sm103 / sm120 are kept stack-specific because they truly differ across CUDA / CUTLASS / CuTe DSL lanes.
   - Used for: CUDA C++/PTX workflow, runtime and driver API reference workflow, kernel templates, profiling, debugging, performance traps, and architecture tuning.

4. **cache-dit CuTe DSL kernel skill**
   - Repository path: <https://github.com/vipshop/cache-dit/tree/main/.copilot/skills/cute-dsl-kernel>
   - Local source path used for the copy: `/Users/bbuf/工作目录/Common/cache-dit/.copilot/skills/cute-dsl-kernel`
   - Copied into: `references/cute-dsl/`
   - The original `SKILL.md` is rewritten as `references/cute-dsl/cute-dsl-overview.md` with all `vipshop/cutlass/...` workspace paths replaced by guidance pointing at an optional external CUTLASS checkout (no CUTLASS source tree is bundled).
   - Used for: CuTe DSL workflow, API reference map, decorators, runtime tensor conversion, pipeline abstractions, nvgpu cpasync, tcgen05, and validation guidance.

5. **cache-dit CUTLASS C++ kernel skill**
   - Repository path: <https://github.com/vipshop/cache-dit/tree/main/.copilot/skills/cutlass-cpp-kernel>
   - Local source path used for the copy: `/Users/bbuf/工作目录/Common/cache-dit/.copilot/skills/cutlass-cpp-kernel`
   - Copied into: `references/cutlass-cpp/`
   - The original `SKILL.md` is rewritten as `references/cutlass-cpp/cutlass-cpp-overview.md` with all `vipshop/cutlass/...` workspace paths replaced by guidance pointing at an optional external CUTLASS checkout.
   - Used for: CUTLASS/CuTe C++ source navigation, collective and epilogue workflow, rewrite validation, architecture-specific profiling, and kernel structure review.

## Restructuring Notes (relative to the original copies)

To make the bundle behave like one natural skill rather than a set of nested skills:

- **All upstream `SKILL.md` files have had their YAML frontmatter (`name`, `description`, `argument-hint`, `user-invocable`) removed**, and have been renamed to `<stack>-overview.md` so no skill loader treats them as standalone invocable skills.
- **Identical files were deduplicated** and promoted to shared top-level locations:
  - `references/kernel-templates.md` (was identical in cuda-cpp / cute-dsl / cutlass-cpp)
  - `references/troubleshooting.md` (was identical in all three)
  - `references/architectures/sm89-optimization-guide.md` (was identical in all three)
  - `references/architectures/sm90-optimization-guide.md` (was identical in all three)
- **Stack-specific files are kept per-stack** because they truly differ:
  - `references/{cuda-cpp,cute-dsl,cutlass-cpp}/sm100-optimization-guide.md`
  - `references/{cuda-cpp,cute-dsl,cutlass-cpp}/sm103-optimization-guide.md`
  - `references/{cuda-cpp,cute-dsl,cutlass-cpp}/sm120-optimization-guide.md`
- **The vendored CUDA / PTX / Profiler documentation tree** that originally lived under cache-dit's `cuda-cpp-kernel/references/` is preserved under `references/cuda-cpp/vendored-docs/` with a top-level README explaining when (and when not) to descend into it.
- **Specialized Triton sub-files** keep their original content but had their per-file frontmatter replaced by an HTML-comment provenance line.

These changes are mechanical reorganizations; no upstream content was paraphrased, summarized, or modified beyond fixing path references inside the moved bundle.
