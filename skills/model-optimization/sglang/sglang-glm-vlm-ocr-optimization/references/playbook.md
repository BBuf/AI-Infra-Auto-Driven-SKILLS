# GLM VLM/OCR Playbook

## Symptom Map

| Symptom | First files | PR trail |
| --- | --- | --- |
| GLM-4.5V startup fails | `glm4v.py`, `glm4v_moe.py`, processor, attention backend | `#8798`, `#9059`, `#9245`, `#9554`, `#17122` |
| Vision encoder DP/PP issue | `glm4v.py`, `glm4v_moe.py`, `general_mm_embed_routine` | `#14097`, `#14720`, `#14927` |
| Transformers field missing | `model_config.py`, `weight_utils.py`, processor registration | `#14998`, `#18885`, `#21134` |
| GLM-OCR load/output issue | `glm_ocr.py`, `glm_ocr_nextn.py`, processor | `#17582`, `#18885`, `#20463`, `#20740`, `#21134` |
| Projection perf/regression | GLM visual patch embedding, shared conv layer | `#20033`, `#20282`, `#20463`, `#20740` |
| Vision RoPE slow or wrong | `rotary_embedding.py`, `glm4v.py` | `#15205`, `#17420` |
| NPU/AMD platform startup | `vision.py`, `fp8_kernel.py`, fused MoE, `glm4_moe.py` | `#15434`, `#19728`, `#22961` |

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

## PR Reading Checklist

- For support PRs such as `#8798` or `#17582`, inspect architecture registration, processor registration, token padding, loader mapping, and tests.
- For performance PRs such as `#15205`, `#17420`, `#20033`, and `#20282`, preserve the before/after hot path and the correctness guard.
- For revert chains such as `#20463` -> `#20740`, document both the discovered failure mode and the current-main revert.
- For open PRs such as `#9349`, `#14662`, `#19728`, and `#22961`, label them as radar only and do not describe them as shipped.

## Change Rules

- Do not patch text GLM MoE as a first move for visual feature bugs.
- GLM-OCR changes need OCR-specific examples.
- Transformers compatibility fixes should detect fields defensively.
- Any doc update must include motivation, implementation idea, key code fragment, validation, and risk/current status for every PR it cites.
