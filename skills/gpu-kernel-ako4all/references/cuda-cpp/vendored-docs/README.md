# Vendored CUDA / PTX / Profiler Documentation

This directory is a verbatim mirror of the documentation tree that
`cache-dit` ships under its `cuda-cpp-kernel` Copilot skill, copied here as
part of `gpu-kernel-ako4all` so the skill is self-contained.

**Source**: https://github.com/vipshop/cache-dit/tree/main/.copilot/skills/cuda-cpp-kernel/references

**Total size**: ~12 MB. Treat as **opt-in reference, not bedtime reading.**

## When to Open Files Here

Only open a file under this tree when you are narrowing to **one specific
question**:

| Question | Where to look |
| --- | --- |
| Exact PTX instruction syntax / semantics | `ptx-docs/9-instruction-set/` |
| Quick PTX cheat sheet | `ptx-simple/` |
| CUDA Runtime API parameter / return-code / module behavior | `cuda-runtime-docs/modules/` or `cuda-runtime.md` |
| CUDA Driver API (context, module loading, virtual memory, tensor maps) | `cuda-driver-docs/modules/` or `cuda-driver.md` |
| Programming-model / execution-model / memory-model behavior | `cuda-guide/` |
| C++ best practices on memory, occupancy, async | `best-practices-guide/` |
| Specific `ncu` section / metric / counter family | `ncu-docs/ProfilingGuide.md`, `ncu-guide.md` |
| Specific `nsys` report / CLI option | `nsys-docs/UserGuide.md`, `nsys-guide.md` |
| `compute-sanitizer` / `cuda-gdb` / `cuobjdump` workflow | `debugging-tools.md`, `../../profiling-debugging-reference.md` |
| Performance traps catalog | `performance-traps.md`, `../../profiling-debugging-reference.md` |
| NVTX usage | `nvtx-patterns.md` |

## When NOT to Open Files Here

- For high-level workflow guidance — use `../cuda-cpp-overview.md` and the
  top-level `../../cuda-cpp-kernel-reference.md` instead.
- For architecture-specific tuning — use `../sm10[0|3].md`, `../sm120-optimization-guide.md`,
  or `../../architectures/sm89-optimization-guide.md` / `sm90-optimization-guide.md`.
- For optimization-loop discipline (AKO harness, iteration log, validation
  gate) — use `../../ako4all-kernel-loop.md`.

## Trimming

If you fork this skill and the 12 MB mirror is too heavy for your repository,
you can safely remove any subdirectory you don't use; the rest of the skill
will keep working. The recommended minimal trim is:

- keep `ncu-docs/`, `nsys-docs/`, `ptx-simple/`, `debugging-tools.md`,
  `performance-traps.md`, `nvtx-patterns.md`
- drop `ptx-docs/` (5.9 MB), `cuda-guide/` (1.6 MB), `cuda-driver-docs/`
  (1.2 MB), `cuda-runtime-docs/` (1.0 MB), `best-practices-guide/` (584 KB)
  if you have NVIDIA documentation locally installed elsewhere.
