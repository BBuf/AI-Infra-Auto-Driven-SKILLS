# GLM-5/5.1 Playbook

## Required PR Reading

Before changing GLM-5/5.1 runtime, docs, quantization, MTP, tool calling, AMD/NPU/GB300 validation, or NSA indexer behavior, read the diff-reviewed cards in `references/pr-history.md`.

- `#18521` and `#18804`: GLM DSA bring-up and fused shared-expert recognition.
- `#20062`: dense MHA versus sparse MLA threshold logic for DSA/GLM on Blackwell.
- `#22285`, `#22336`, and `#22399`: H200/B200/AMD/GB300 validation lanes.
- `#22314`, `#22543`, `#22850`, and `#23219`: FP8 KV, Quark/MXFP4, AITER indexer, and GLM-5-MXFP4 MTP implementation details.
- `#22595`: GLM-5.1 tool-result normalization in the OpenAI chat path.

## Symptom Map

| Symptom | First files | PR trail |
| --- | --- | --- |
| GLM-5 startup or loading fails | `glm4_moe.py`, config detection | `#18521`, `#18804`, `#22285` |
| FP8/MXFP4/NVFP4 failure | quant loader, AMD/GB300 tests | `#21710`, `#21773`, `#22336`, `#22543` |
| MTP/NextN bug | `deepseek_nextn.py`, `glm4_moe_nextn.py` | `#23219` |
| NSA indexer perf issue | `nsa_indexer.py`, AITER path | `#22850` |
| Tool template wrong | `serving_chat.py`, tool normalization tests | `#22595` |
| Dense-attention threshold unexpected | `server_args.py`, `nsa_backend.py`, `environ.py` | `#20062` |

## Investigation Order

1. Record dense/NSA backend and KV dtype.
2. Verify base FP8/BF16 startup.
3. Validate quantized checkpoint loading.
4. Enable MTP/NextN.
5. Validate chat template and tool-result behavior.
6. Profile NSA indexer and backend.
7. Add PCG/EAGLE/CUDA graph only after the base DSA and MTP lanes pass.

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
- If the patch touches `deepseek_v2.py`, `nsa_backend.py`, or `nsa_indexer.py`, state whether the behavior is GLM-specific or shared with DeepSeek V3.2.
- Tool-template fixes need OpenAI-compatible chat completion tests.
- NSA changes need BF16 and quantized KV validation.
- Do not rename GLM-5 NVFP4 paths to GLM-5.1 unless a real GLM-5.1 NVFP4 checkpoint exists.
