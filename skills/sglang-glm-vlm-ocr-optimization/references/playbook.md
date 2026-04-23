# GLM VLM/OCR Playbook

## Symptom Map

| Symptom | First files | PR trail |
| --- | --- | --- |
| GLM-4.5V startup fails | `glm4v.py`, `glm4v_moe.py`, processor | `#8798`, `#9059`, `#9245`, `#9554` |
| Vision encoder DP issue | `glm4v_moe.py`, vision DP path | `#14097`, `#14720`, `#14927` |
| Transformers field missing | config/processor registration | `#14998`, `#18885`, `#21134` |
| GLM-OCR load/output issue | `glm_ocr.py`, `glm_ocr_nextn.py` | `#17582`, `#20463`, `#20740`, `#21134` |
| Projection perf issue | visual projection path | `#20033` |

## Investigation Order

1. Validate processor registration.
2. Check feature tensor shape and dtype.
3. Run no-DP baseline.
4. Enable encoder DP/PP.
5. Validate quantized startup.
6. Run OCR/Glyph task examples.

## Official Deployment Checklist

- Launch flags: FP8/BF16 checkpoint, `--keep-mm-feature-on-device`, `--mm-attention-backend`, `--mm-max-concurrent-calls`, and `--mm-enable-dp-encoder`.
- Transport/cache: record `SGLANG_USE_CUDA_IPC_TRANSPORT=1` and `SGLANG_VLM_CACHE_SIZE_MB=0` when reproducing feature-transfer or cache issues.
- Thinking budget: GLM VLM thinking-budget tests use the same custom logit processor path as GLM text.
- GLM-Glyph belongs in this lane because it shares multimodal/OCR cookbook and processor surfaces.

## Validation Checklist

- one image prompt
- OCR page prompt
- encoder DP and no-DP comparison
- transformers compatibility path
- quantized startup if loader changed
- NPU/AMD if platform code changed

## Change Rules

- Do not patch text GLM MoE as a first move for visual feature bugs.
- GLM-OCR changes need OCR-specific examples.
- Transformers compatibility fixes should detect fields defensively.
