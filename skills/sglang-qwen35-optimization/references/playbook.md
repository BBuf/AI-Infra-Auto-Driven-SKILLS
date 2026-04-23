# Qwen3.5 Playbook

## Symptom Map

| Symptom | First files | PR trail |
| --- | --- | --- |
| Load failure in PP or quantized checkpoint | `qwen3_5.py`, `qwen3_5_mtp.py`, loaders | `#18489`, `#18937`, `#19670`, `#21070`, `#21347`, `#21448` |
| Dense model precision mismatch | `qwen3_5.py`, linear attention precision | `#19070`, `#19961` |
| MTP + EPLB failure | `qwen3_5_mtp.py`, EPLB/MoE paths | `#19767`, `#23147`, `#23462` |
| GDN projection overhead | `qwen3_5.py` projection blocks | `#20386`, `#21019`, `#22312` |
| AMD quant/perf gap | AITER, AMD tests, MoE fusion | `#20736`, `#21234`, `#21669`, `#23146` |

## Investigation Order

1. Reproduce with BF16 and no MTP.
2. Reproduce with target quantization and no MTP.
3. Enable PP and tied embeddings if the deployment uses PP.
4. Enable MTP and EPLB after base loading is proven.
5. Profile GDN and shared expert kernels only after logits match.

## Official/AMD Deployment Checklist

- AMD path: record `--attention-backend triton`, `SGLANG_USE_AITER=1`, AITER MoE backend, and ROCm architecture.
- Parser path: official docs use `--reasoning-parser qwen3`; tool-use examples use `--tool-call-parser qwen3_coder`.
- Hybrid path: do not mix Mamba cache, GDN fusion, shared-expert fusion, MTP, and quantization in the first repro.
- Multimodal variants: include DeepStack Vision/Conv3d request coverage when the checkpoint is not text-only.
- FP8 KV: keep the accuracy caution in command docs and benchmark notes.

## Validation Commands

```bash
python -m pytest test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py
python -m pytest test/registered/4-gpu-models/test_qwen35_fp4_triton.py
python -m pytest test/registered/8-gpu-models/test_qwen35.py
python -m pytest test/registered/gb300/test_qwen35_fp8.py
python -m pytest test/registered/gb300/test_qwen35_nvfp4.py
```

## Change Rules

- Loader fixes should name the exact checkpoint/exporter they support.
- PP fixes must cover skipped/missing layer handling and tied embeddings.
- MTP fixes must compare target and draft paths.
- Cookbook updates should mention FP8 KV accuracy caveats when relevant.
