# GLM-4.5 Playbook

## Symptom Map

| Symptom | First files | PR trail |
| --- | --- | --- |
| A2A/MoE startup failure | `glm4_moe.py`, A2A backend init | `#11692`, `#11800` |
| Reduce-scatter regression | MoE block and server flags | `#11665` |
| NVFP4 scale shape issue | quant loader and scale padding | `#12497` |
| Tool stream interruption | `glm4_moe_detector.py` | `#17714`, `#23067` |
| Compressed-tensors load gap | `glm4_moe_lite.py`, quant configs | `#19106` |

## Investigation Order

1. Reproduce BF16 with minimal TP/EP.
2. Enable A2A/DeepEP and reduce-scatter exactly as the deployment uses it.
3. Reproduce quantized loading.
4. Run parser streaming.
5. Profile MoE kernels only after correctness is stable.

## Official Deployment Checklist

- Parser split: GLM-4.5 uses `glm45`; GLM-4.7 uses `glm47`.
- Thinking budget: use `--enable-custom-logit-processor` when testing GLM thinking-budget behavior.
- Speculative path: keep MTP/EAGLE tests separate from base MoE startup.
- Public-doc baseline: GLM-4.5 has native function calling and long-context deployment expectations; parser tests should cover streaming and non-streaming.

## Validation Checklist

- BF16 smoke.
- FP8/NVFP4 startup.
- A2A/DeepEP with MoE.
- Tool parser streaming.
- Cookbook command parity.

## Change Rules

- Keep VLM fixes out of GLM-4.5 text skill unless the text MoE class is the source.
- Do not change reduce-scatter defaults without a hardware-specific benchmark.
- Parser fixes need streaming and non-streaming tests.
