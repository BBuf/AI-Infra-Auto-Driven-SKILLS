# GLM-5/5.1 Playbook

## Symptom Map

| Symptom | First files | PR trail |
| --- | --- | --- |
| GLM-5 startup or loading fails | `glm4_moe.py`, config detection | `#18521`, `#18804`, `#22285` |
| FP8/MXFP4/NVFP4 failure | quant loader, AMD/GB300 tests | `#21710`, `#21773`, `#22336`, `#22543` |
| MTP/NextN bug | `glm4_moe_nextn.py`, `deepseek_nextn.py` | `#23219`, `#22823`, `#23037` |
| NSA indexer perf issue | `nsa/` indexer/backend files | `#22850`, `#22851`, `#23351` |
| Tool template wrong | chat template/tool normalization | `#22595`, `#22977` |
| Dense-attention threshold unexpected | server args/env and DSA backend | `#20062`, `#22473` |

## Investigation Order

1. Record dense/NSA backend and KV dtype.
2. Verify base FP8/BF16 startup.
3. Validate quantized checkpoint loading.
4. Enable MTP/NextN.
5. Validate chat template and tool-result behavior.
6. Profile NSA indexer and backend.
7. Add PCG/EAGLE/CUDA graph.

## Official Deployment Checklist

- Launch: for GLM-5, official docs use the DeepSeek V3.2/DSA-style command with the model path replaced by `zai-org/GLM-5-FP8`.
- Parsers: `--tool-call-parser glm47` and `--reasoning-parser glm45`.
- NSA: record prefill/decode backend, `--nsa-prefill-cp-mode`, `--page-size 64`, and dense-attention fallback threshold.
- HiSparse: validate `--enable-hisparse` and `--hisparse-config` separately for GLM-5.1 long-context/high-concurrency work.
- FP8 KV: keep accuracy caveats visible in benchmark and cookbook updates.

## Validation Commands

```bash
python -m pytest test/registered/8-gpu-models/test_glm_51_fp8.py
python -m pytest test/registered/gb300/test_glm5_fp8.py
python -m pytest test/registered/gb300/test_glm5_nvfp4.py
```

## Change Rules

- If the patch touches `deepseek_nextn.py`, mention GLM-5 and DeepSeek V3.2 shared MTP impact.
- Tool-template fixes need OpenAI-compatible chat completion tests.
- NSA changes need BF16 and quantized KV validation.
