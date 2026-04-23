# Qwen3-Next Playbook

## Symptom Map

| Symptom | First files | PR trail |
| --- | --- | --- |
| Weight loading fails for FP8/NVFP4 | `qwen3_next.py`, loader helpers | `#18224`, `#21313`, `#21496`, `#21662` |
| MTP hangs or mismatches | `qwen3_next_mtp.py`, speculative workers | `#22458`, `#20397`, `#19812` |
| GDN projection too slow | `qwen3_next.py`, GDN projection kernels | `#19321`, `#21019` |
| FlashInfer all-reduce not used | all-reduce/fusion dispatch | `#22664` |
| CPU offload or cache issue | mamba/radix scheduler and hybrid model cache | `#23474`, `#12892` |

## Investigation Order

1. Reproduce without MTP and with `--mamba-scheduler-strategy no_buffer`.
2. Reproduce with target quantization but without speculative decoding.
3. Add `extra_buffer` only after baseline state behavior is correct.
4. Enable MTP and verify draft-layer cache sharding.
5. Only then profile GDN projection fusion and all-reduce.

## Official Flag Checklist

- Mamba/radix state: `--max-mamba-cache-size`, `--mamba-ssm-dtype`, `--mamba-full-memory-ratio`.
- Scheduler: `--mamba-scheduler-strategy extra_buffer --page-size 64`.
- Speculative decoding: keep NEXTN/EAGLE flags separate from base Mamba state validation.
- Parsers: official docs use `--tool-call-parser qwen` and `--reasoning-parser qwen3`.
- Long context: record YaRN/context-length overrides separately from cache and scheduler changes.

## Validation Commands

```bash
python -m pytest test/registered/4-gpu-models/test_qwen3_next_models.py
python -m pytest test/registered/4-gpu-models/test_qwen3_next_models_mtp.py
python -m pytest test/registered/models/test_qwen3_next_models_fp4.py
```

For cookbook parity, compare generated commands in:

- `docs_new/src/snippets/autoregressive/qwen3-next-deployment.jsx`
- `sgl-cookbook/src/components/autoregressive/Qwen3NextConfigGenerator/`

## Change Rules

- Avoid hiding Qwen3-Next fixes in generic Qwen3 code unless the bug is demonstrably shared.
- Keep MTP changes covered by target/draft and no-MTP tests.
- Weight-loader changes need at least one quantized checkpoint test.
- Any scheduler change should be tested with prefix/radix cache reuse.
