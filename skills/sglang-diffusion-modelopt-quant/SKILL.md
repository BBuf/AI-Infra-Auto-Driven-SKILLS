---
name: sglang-diffusion-modelopt-quant
description: Use when quantizing a diffusion DiT with NVIDIA ModelOpt and making the resulting FP8 or NVFP4 checkpoint loadable, verifiable, and benchmarkable in SGLang Diffusion. This is the right skill for ModelOpt export conversion, BF16-vs-quant validation, multi-transformer override wiring, and remote GPU bring-up with skills such as `rtx5090`.
---

# SGLang Diffusion ModelOpt Quant

## Overview

Use this skill when the task is to take a diffusion transformer through the full ModelOpt workflow: quantize it, adapt the export to SGLang, verify that quality holds up, and check whether the quantized checkpoint is actually faster.

Run commands from the SGLang repo root unless the task explicitly says otherwise. When the work needs real CUDA validation, pair this skill with a remote GPU skill such as `rtx5090` or `h100-sglang-diffusion`.

## Core Rules

- Use ModelOpt's official diffusion quantization entrypoint as the PTQ source of truth.
- Keep the workflow generic. Put model-specific fallback logic in small isolated branches, not in the main conversion path.
- Benchmark only when the BF16 and quantized commands are identical except for the checkpoint override being tested.
- For diffusion FP8, pin `dit_cpu_offload=false` and `dit_layerwise_offload=false`.
- For multi-transformer pipelines, prefer per-component overrides when different backbones need different checkpoints.
- If the active SGLang branch does not yet contain the expected conversion or validation tool, start from this skill's bundled `scripts/` copy and add it under `python/sglang/multimodal_gen/tools/` instead of inventing one-off scripts elsewhere.

## Read First

Read these sources before changing code:

- NVIDIA ModelOpt diffusion guide:
  - `examples/diffusers/README.md`
  - `examples/diffusers/quantization/quantize.py`
  - `examples/diffusers/quantization/config.py`
- SGLang diffusion quantization and loading paths:
  - `python/sglang/multimodal_gen/runtime/layers/quantization/modelopt_quant.py`
  - `python/sglang/multimodal_gen/runtime/utils/quantization_utils.py`
  - `python/sglang/multimodal_gen/runtime/loader/transformer_load_utils.py`
- Bundled helper scripts in this skill:
  - `scripts/convert_modelopt_fp8_checkpoint.py`
  - `scripts/compare_diffusion_trajectory_similarity.py`

If you are working on a new model family, inspect the model's transformer config and tensor naming before changing the generic converter.

## What This Skill Owns

This skill is for the ModelOpt-to-SGLang bridge:

- quantizing a diffusion DiT with ModelOpt
- converting or adapting the export so SGLang can load it
- validating BF16 versus quantized quality with fixed prompt and seed
- benchmarking under matched settings
- adding or refining model-specific fallback logic only when the generic path is insufficient

It is not a general kernel-tuning or diffusion architecture design skill.

## FP8 Vs NVFP4

FP8 and NVFP4 usually need different treatment.

FP8:
- the validated ModelOpt diffusion export typically still needs an extra SGLang-side conversion step
- SGLang expects explicit `weight_scale` and `input_scale`
- the validated path usually materializes SGLang-native `float8_e4m3fn` weights from `backbone.pt`

NVFP4:
- the export often already contains packed FP4 weights, scale tensors, and enough metadata for SGLang to reconstruct the quant config
- in that case SGLang mainly needs the right loader detection and tensor layout handling
- packed-QKV families still need special care

"Often" does not mean "always". The exact load path depends on the checkpoint family and tensor layout.

## Workflow

### 1. Verify The BF16 Baseline First

Before quantizing anything:

- run the original BF16 model in SGLang
- fix the prompt, seed, size, step count, and GPU topology
- save the output plus timing or `perf.json`

Do not start ModelOpt work until the BF16 path is already healthy.

### 2. Quantize With Official ModelOpt

Generic template:

```bash
python quantize.py \
  --model <model-name> \
  --override-model-path <hf-repo-or-local-model> \
  --model-dtype <Half|BFloat16> \
  --format <fp8|nvfp4> \
  --batch-size 1 \
  --calib-size <calib-size> \
  --n-steps <calib-steps> \
  --quantize-mha \
  --prompts-file <prompt-file> \
  --quantized-torch-ckpt-save-path <out>/ckpt \
  --hf-ckpt-dir <out>/hf
```

For multi-transformer models:
- quantize each backbone deliberately
- keep each output directory separate
- save both `backbone.pt` and the matching `hf/<component>` export

### 3. Convert FP8 Exports For SGLang

FP8 usually needs an extra conversion step:

```bash
PYTHONPATH=python python3 -m sglang.multimodal_gen.tools.convert_modelopt_fp8_checkpoint \
  --modelopt-hf-dir <out>/hf \
  --modelopt-backbone-ckpt <out>/ckpt/backbone.pt \
  --base-transformer-dir <base-model-transformer-dir> \
  --output-dir <out>/sglang_transformer \
  --overwrite
```

The converter should:
- read `weight_quantizer._amax` and `input_quantizer._amax` from `backbone.pt`
- write `weight_scale` and `input_scale`
- materialize eligible FP8 weights as `float8_e4m3fn`
- preserve ModelOpt `ignore` layers as BF16

For multi-transformer pipelines, run the converter once per exported backbone.

### 4. Load The Quantized Checkpoint In SGLang

Single-transformer example:

```bash
sglang generate \
  --model-path <base-model> \
  --transformer-path <quantized-transformer> \
  --prompt "<prompt>" \
  --seed <seed> \
  --save-output
```

Multi-transformer example:

```bash
sglang generate \
  --model-path <base-model> \
  --transformer-path <quantized-transformer> \
  --transformer-2-path <another-transformer-or-bf16-override> \
  --prompt "<prompt>" \
  --seed <seed> \
  --save-output
```

Guidelines:
- use global `--transformer-path` only when one transformer override is enough
- use `--<component>-path` when components differ
- if a config-expanded form also works, keep the CLI readable in examples

### 5. Validate Accuracy

Use two levels of validation.

Reduced deterministic validation:
- keep prompt, seed, resolution, and step count fixed
- compare BF16 and quantized runs
- capture denoising trajectories
- inspect per-step latent cosine similarity plus MAE or RMSE
- compare final frames with image metrics such as PSNR or MAE

Tool:

```bash
PYTHONPATH=python python3 -m sglang.multimodal_gen.tools.compare_diffusion_trajectory_similarity \
  --model-path <base-model> \
  --prompt "<prompt>" \
  --width <w> \
  --height <h> \
  --num-inference-steps <steps> \
  --guidance-scale <cfg> \
  --seed <seed> \
  --candidate-transformer-path <quantized-transformer> \
  --output-json <report.json>
```

Full-output validation:
- run the real user-facing generation config in both BF16 and quantized mode
- inspect images or representative video frames visually
- only claim "quality preserved" for the exact scope you actually checked

### 6. Benchmark Correctly

Benchmark only when these match between BF16 and quantized runs:

- prompt
- seed
- width and height
- frame count
- inference step count
- GPU count and topology
- offload flags
- compile settings
- profiler settings

Only the checkpoint override should differ.

Interpretation rule:
- the main expected gain is usually in denoising
- do not over-attribute wins in unrelated stages unless those components were also quantized

### 7. Add Model-Specific Fallbacks Only When Needed

If the generic FP8 path fails on a new model family:

- inspect which modules are numerically sensitive or loader-incompatible
- keep fallback patterns small and explicit
- isolate them in the converter instead of scattering ad-hoc exceptions
- re-run deterministic trajectory checks after every fallback change

Do not turn one validated model quirk into a generic rule unless another family also needs it.

## FP8 Offload Constraint

Current diffusion ModelOpt FP8 support requires:

- `dit_cpu_offload=false`
- `dit_layerwise_offload=false`

Reason:
- the FP8 linear path depends on a CUTLASS-compatible weight layout after loading
- offload and restore paths do not reliably preserve that layout
- layerwise offload in particular can rebuild weights in a way that breaks the column-major contract expected by the FP8 GEMM path

Keep those flags pinned explicitly in benchmark commands even if the runtime also guards them.

## Reference Models

Useful validated references:

- `black-forest-labs/FLUX.2-dev`
- `Wan-AI/Wan2.2-T2V-A14B-Diffusers`

Use them as sanity checks for workflow shape, not as proof that a new model family needs the same fallback profile.

## Current Code Areas

These are the first places to inspect when wiring a new quantized diffusion model:

- `python/sglang/multimodal_gen/runtime/layers/quantization/__init__.py`
- `python/sglang/multimodal_gen/runtime/layers/quantization/modelopt_quant.py`
- `python/sglang/multimodal_gen/runtime/utils/quantization_utils.py`
- `python/sglang/multimodal_gen/runtime/loader/transformer_load_utils.py`
- `python/sglang/multimodal_gen/runtime/models/dits/`
- `python/sglang/multimodal_gen/tools/convert_modelopt_fp8_checkpoint.py`
- `python/sglang/multimodal_gen/tools/compare_diffusion_trajectory_similarity.py`

## Claim Discipline

When documenting results:

- claim only scopes that were actually validated end to end
- do not collapse "primary-transformer FP8" into "full-model FP8"
- do not call it a fair benchmark if BF16 and quantized runs used different offload behavior
