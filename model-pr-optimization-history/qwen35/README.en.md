# SGLang Qwen3.5 Support and Optimization Timeline

Evidence snapshot: SGLang `origin/main` `b3e6cf60a` on 2026-04-22 and sgl-cookbook `origin/main` `816bad5` on 2026-04-21.

Scope: Qwen3.5 dense/MoE, MTP, FP8, NVFP4, MXFP4, PP/EPLB, GDN fusion, AMD/NPU/Blackwell deployment.

## Summary

Qwen3.5 is one of the most active Qwen text lanes. The main risks are weight loading, PP layer split, tied embeddings, MTP + EPLB, GDN projection fusion, shared expert fusion, quantization guards, and cookbook command sync for H200/B200/AMD/GB300.

## Code Surfaces

- `python/sglang/srt/models/qwen3_5.py`
- `python/sglang/srt/models/qwen3_5_mtp.py`
- `python/sglang/srt/configs/qwen3_5.py`
- `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`
- `test/registered/4-gpu-models/test_qwen35_fp4_triton.py`
- `test/registered/8-gpu-models/test_qwen35.py`
- `test/registered/gb300/test_qwen35_fp8.py`
- `test/registered/gb300/test_qwen35_nvfp4.py`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx`

## Merged PRs

Key merged PRs: `#18489`, `#18538`, `#18937`, `#19070`, `#19220`, `#19411`, `#19670`, `#19767`, `#19889`, `#19961`, `#20386`, `#20736`, `#21019`, `#21070`, `#21234`, `#21347`, `#21448`, `#21669`, `#22312`, `#22913`.

## Open Radar

Track `#19484`, `#20074`, `#20565`, `#21668`, `#22618`, `#23146`, `#23147`, `#23331`, and `#23462`.

## Cookbook Evidence

Track `sgl-cookbook#164`, `#168`, `#169`, `#177`, `#179`, `#207`, `#214`, `#230`, and `#237`.

## Next Work

Turn PP/MTP/EPLB/quantization into a fixed regression matrix, validate GDN/DeltaNet and AMD EAGLE work against BF16 and quantized baselines, and keep FP8 KV caution notes synchronized with server defaults.

## 2026-04-23 Three-Pass Addendum

Detailed ledger: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`. This pass adds `#18926`, `#19391`, `#20864`, `#21487`, `#21849`, `#22145`, `#22240`, `#22431`, `#22493`, `#22908`, and `#22948`.

Additional open radar: `#19585`, `#19781`, `#19956`, `#19974`, `#20029`, `#20255`, `#20370`, `#20448`, `#20667`, `#20789`, `#20918`, `#21185`, `#21217`, `#21464`, `#22027`, `#22325`, `#22502`, `#22867`, `#22885`, `#23062`, `#23468`, and `#23471`. Public evidence adds the official SGLang Qwen3.5 docs and AMD day-0 article for AMD/Triton/GDN/shared-expert/vision-kernel coverage.
