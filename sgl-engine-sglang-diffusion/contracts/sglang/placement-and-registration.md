# SGLang placement and registration contract

This contract is interpreted against the campaign's locked SGLang commit. The
controller must inspect that checkout and fail closed if it recognizes neither
the current unified kernel tree nor the legacy JIT tree. Host worktrees and
unpinned `main` are not placement authorities.

## Current unified kernel tree

For a generated model slug `<model>`, keep agent-owned artifacts isolated:

```text
python/sglang/kernels/agent/diffusion/<model>/         profile, dispatch, manifest, receipts
python/sglang/kernels/ops/diffusion/agent/<model>/     callable Python wrappers
python/sglang/kernels/jit/csrc/diffusion/agent/<model>/ lightweight JIT CUDA sources
python/sglang/kernels/aot/csrc/diffusion/agent/<model>/ heavyweight AOT sources
python/sglang/kernels/aot/include/diffusion/agent/<model>/ AOT headers
python/sglang/kernels/aot/python/sgl_kernel/diffusion/agent/<model>/ AOT Python implementation
test/registered/kernels/ops/diffusion/agent/<model>/   GPU correctness tests
test/registered/kernels/benchmark/diffusion/agent/<model>/ focused benchmarks
```

Runtime code imports callable kernels through `sglang.kernels.ops.*`.
`sglang.kernels.agent` owns profile selection and observability, not a second
public operator namespace.

Registration follows the unified-kernel RFC:

- `KernelSpec` registration records metadata only; it must not import torch,
  load a backend, or compile during registration.
- Multiple registered backends form an inventory. They do not imply a hidden
  priority order.
- A multi-backend fused operator supplies a pure native reference and uses
  `BaseFusedOp` eligibility and fallback behavior.
- A profile-selected model fast path still needs exact model, source, hardware,
  shape, dtype, and feature guards.

## Lightweight JIT implementation

Prefer JIT for a self-contained CUDA/C++ kernel without a large dependency or
wheel requirement. Use SGLang's JIT abstractions rather than hand-written
loader plumbing: symbolic tensor matching, runtime guards, cached compilation,
the canonical launch wrapper, and thin Python argument marshalling. Sources
remain relative to the checked-in JIT `csrc` root. Add parity coverage,
unsupported-shape fallback coverage, and a registered benchmark.

## Heavyweight AOT implementation

Use the locked tree's AOT root for CUTLASS, large dependencies, or code that
must ship in the prebuilt kernel wheel. A complete operator change includes:

1. CUDA/C++ implementation and declaration;
2. torch operator registration in the common extension;
3. deterministically sorted CMake source registration;
4. Python wrapper and package export;
5. correctness tests and benchmarks; and
6. an AOT build/import validation.

The generated implementation remains in the model-scoped agent subtree while
its declarations, extension registration, build entries, and canonical
`sglang.kernels.ops` wrapper use the normal AOT integration points.

## Runtime and evidence

The final patch exposes:

```text
--quality off
--quality auto
--quality <profile-id>
```

`off` is identical to the locked native source. `auto` activates only an exact
immutable profile match. Unsupported calls follow the profile's explicit
native-fallback or hard-error policy. Every inference records selected profile,
operator/backend engagement counts, fallback counts/reasons, and source hashes.

The native SGLang backend is mandatory for optimized measurements. Logs
containing a Diffusers fallback marker invalidate the run. Warmup, topology,
model inputs, seeds, precision, denoising steps, and timing scope stay frozen.
Microbenchmarks and profiles guide search; only the complete frozen workload
can establish campaign speedup.

## Correctness precedence

These rules add implementation placement and optimization knowledge. They do
not weaken the locked Sol-Engine correctness contract:

- lossless residency/kernel work must preserve logical denoising work and pass
  independent method/code audit;
- cache, PISA, quantization, and token-pruning work must pass aligned LPIPS and
  independent built-in-vision review on the complete five-prompt workload.
