# Quantization work-order scope

## Adaptation status and identity

`quantization` brings Sol-Engine's full quantization technique family into a
standalone SGLang work-order lane. Sol-Engine's paper and technique
documentation contain this family, but its reviewed lightweight registry does
not assign it a standalone lane. The **20-round hard budget is an SGLang
adaptation default**, not an upstream lightweight setting.

This is a quality-gated lane. It may explore selective low-precision weights,
activations, attention, and linear paths, including NVFP4, diffusion-aware PTQ,
SVDQuant/Nunchaku, SageAttention, and ModelOpt FP8. Uniform low-bit conversion
without profiling and sensitivity evidence is not the objective.

## Search and ownership

Run a hardware/runtime preflight before consuming candidate rounds. Record GPU
architecture, CUDA and library versions, checkpoint format, native low-precision
kernel availability, calibration provenance, and an OFF path that restores the
source-current dtype behavior. Unsupported hardware or a dense fallback is a
blocker/diagnostic, not a low-precision performance result.

Search axes may include profiled module scope, weight/activation precision,
layer and denoising-step dense guards, timestep-dependent scaling, backend and
padding, calibration, recipe flags, and fused epilogues. Early and late
denoising steps and model boundary layers are expected sensitivity probes, not
preselected answers.

Do not claim unrelated cache, PISA, token-pruning, topology, scheduler,
step-count, resolution, or prompt gains. Kernel work may implement the
quantized primitive, but the quality/performance claim and activation remain
owned by this lane.

## Required full-workload gate

Every scored point runs the complete frozen five-prompt workload and preserves
model, checkpoint identity or immutable derived-checkpoint identity, VAE,
scheduler, prompts, seeds, resolution, frames, steps, guidance, decode,
hardware, and timing scope. Retention requires:

- positive quantized module/kernel dispatch and zero disallowed silent fallback;
- exact module list, bitwidths, scaling/calibration recipe, dense guards,
  backend, and artifact/checkpoint SHA-256;
- OFF identity, source hashes, run provenance, benchmark and memory evidence;
- aligned prompt-level LPIPS; and
- a passing built-in multimodal visual verdict with no external vision API.

The visual review covers temporal and spatial artifacts, high-frequency detail,
faces/hands/text when present, hallucination, shimmer, snow, flicker, and dense
guard failures. A nominal dtype flag, unavailable native kernel, low LPIPS
alone, or speed inferred from a GEMM microbenchmark cannot pass.

Derived weights must be available through an immutable revisioned artifact with
size and SHA-256. A candidate whose required weights cannot accompany the
patch's reproducible activation remains experimental and cannot be the claimed
patch-only delivery.
