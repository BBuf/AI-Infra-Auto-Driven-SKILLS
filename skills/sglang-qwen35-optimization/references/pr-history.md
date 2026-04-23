# Qwen3.5 PR History

Evidence sweep:

- SGLang `origin/main`: `b3e6cf60a` (`2026-04-22`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- Searched paths: `qwen3_5.py`, `qwen3_5_mtp.py`, Qwen3.5 config, Qwen3.5 tests, cookbook snippets/docs.
- Searched PR terms: `Qwen3.5`, `qwen3.5`, `qwen35`, `qwen3_5`, `Qwen 3.5`.

## Runtime Surfaces

- `python/sglang/srt/models/qwen3_5.py`
- `python/sglang/srt/models/qwen3_5_mtp.py`
- `python/sglang/srt/configs/qwen3_5.py`
- `docs/basic_usage/qwen3_5.md`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3.5.mdx`
- `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx`
- `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`
- `test/registered/4-gpu-models/test_qwen35_fp4_triton.py`
- `test/registered/8-gpu-models/test_qwen35.py`
- `test/registered/gb300/test_qwen35_fp8.py`
- `test/registered/gb300/test_qwen35_nvfp4.py`

## Merged/Current-Main PRs

- [#18489](https://github.com/sgl-project/sglang/pull/18489): initial Qwen3.5 model support.
- [#18538](https://github.com/sgl-project/sglang/pull/18538): refactor `Qwen3_5ForCausalLMMTP`.
- [#18544](https://github.com/sgl-project/sglang/pull/18544): Qwen3.5 follow-up.
- [#18937](https://github.com/sgl-project/sglang/pull/18937): NVFP4 checkpoint support.
- [#19070](https://github.com/sgl-project/sglang/pull/19070): dense Qwen3.5 TP>1 precision bug.
- [#19220](https://github.com/sgl-project/sglang/pull/19220): PCG fix for Qwen3.5.
- [#19411](https://github.com/sgl-project/sglang/pull/19411): Qwen3.5-27B repeat bug.
- [#19670](https://github.com/sgl-project/sglang/pull/19670): pipeline parallel support.
- [#19767](https://github.com/sgl-project/sglang/pull/19767): Qwen3.5 MTP and EPLB.
- [#19889](https://github.com/sgl-project/sglang/pull/19889): TRTLLM all-reduce fusion.
- [#19961](https://github.com/sgl-project/sglang/pull/19961): linear-attention `a_log` precision to FP32.
- [#20386](https://github.com/sgl-project/sglang/pull/20386): replace `einops.rearrange` with flatten in GDN path.
- [#20736](https://github.com/sgl-project/sglang/pull/20736): AMD shared-expert fusion with router experts for BF16/FP8.
- [#21019](https://github.com/sgl-project/sglang/pull/21019): fuse split/reshape/cat ops in GDN projection.
- [#21070](https://github.com/sgl-project/sglang/pull/21070): PP layer splitting fix.
- [#21234](https://github.com/sgl-project/sglang/pull/21234): AMD MXFP4 Qwen3.5-397B support.
- [#21347](https://github.com/sgl-project/sglang/pull/21347): PP tied-embedding weight loading for 4B dense.
- [#21448](https://github.com/sgl-project/sglang/pull/21448): MoE loading and Mamba cache sharding in PP.
- [#21669](https://github.com/sgl-project/sglang/pull/21669): AMD Qwen3.5-397B FP8 nightly performance.
- [#21692](https://github.com/sgl-project/sglang/pull/21692): Qwen3.5 follow-up touching model path.
- [#22312](https://github.com/sgl-project/sglang/pull/22312): non-contiguous tensor support for GDN B/A tensors.
- [#22358](https://github.com/sgl-project/sglang/pull/22358): DFLASH support across Qwen paths.
- [#22913](https://github.com/sgl-project/sglang/pull/22913): split Qwen3.5 tests and update partitions.

## Open PR Radar

- [#19484](https://github.com/sgl-project/sglang/pull/19484): CPU optimization.
- [#20074](https://github.com/sgl-project/sglang/pull/20074): DeltaNet input projection fusion.
- [#20565](https://github.com/sgl-project/sglang/pull/20565): H200 TP4 fused MoE Triton config.
- [#21668](https://github.com/sgl-project/sglang/pull/21668): Intel XPU enablement.
- [#22618](https://github.com/sgl-project/sglang/pull/22618): compressed-tensors NVFP4 guard.
- [#23146](https://github.com/sgl-project/sglang/pull/23146): AMD EAGLE speculative FP8/MXFP4 AITER.
- [#23147](https://github.com/sgl-project/sglang/pull/23147): random MTP initialization.
- [#23331](https://github.com/sgl-project/sglang/pull/23331): adaptive speculative conflicts for hybrid GDN.
- [#23462](https://github.com/sgl-project/sglang/pull/23462): NPU MTP support.

## Cookbook Evidence

- sgl-cookbook [#164](https://github.com/sgl-project/sgl-cookbook/pull/164): initial Qwen3.5 cookbook.
- sgl-cookbook [#168](https://github.com/sgl-project/sgl-cookbook/pull/168): FP8 and NVFP4.
- sgl-cookbook [#169](https://github.com/sgl-project/sgl-cookbook/pull/169): B200 configs.
- sgl-cookbook [#177](https://github.com/sgl-project/sgl-cookbook/pull/177): H200 FP8 optimization with TP=8/EP=8.
- sgl-cookbook [#179](https://github.com/sgl-project/sgl-cookbook/pull/179): AMD MI300X/MI325X/MI355X support.
- sgl-cookbook [#180](https://github.com/sgl-project/sgl-cookbook/pull/180): more Qwen3.5 variants.
- sgl-cookbook [#207](https://github.com/sgl-project/sgl-cookbook/pull/207): B200 FP8 FlashInfer all-reduce fusion tip.
- sgl-cookbook [#214](https://github.com/sgl-project/sgl-cookbook/pull/214): H200 FP8 MTP deployment command.
- sgl-cookbook [#230](https://github.com/sgl-project/sgl-cookbook/pull/230): B200 FP4/NVFP4 generator update.
- sgl-cookbook [#237](https://github.com/sgl-project/sgl-cookbook/pull/237): FP8 KV accuracy caution and command generation cleanup.

## Validation Notes

- Treat Qwen3.5 PP, MTP, EPLB, and quantization as a combined compatibility matrix.
- GB300 tests are separate because Blackwell FP8/NVFP4 behavior can differ from H200.
- Cookbook recipes are actively updated; compare both `sglang/docs_new` and `sgl-cookbook` before publishing command changes.

## Three-Pass Completeness Addendum (2026-04-23)

Full audit ledger: `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`.

Additional merged/current-main PRs: [#18926](https://github.com/sgl-project/sglang/pull/18926), [#19391](https://github.com/sgl-project/sglang/pull/19391), [#20864](https://github.com/sgl-project/sglang/pull/20864), [#21487](https://github.com/sgl-project/sglang/pull/21487), [#21849](https://github.com/sgl-project/sglang/pull/21849), [#22145](https://github.com/sgl-project/sglang/pull/22145), [#22240](https://github.com/sgl-project/sglang/pull/22240), [#22431](https://github.com/sgl-project/sglang/pull/22431), [#22493](https://github.com/sgl-project/sglang/pull/22493), [#22908](https://github.com/sgl-project/sglang/pull/22908), [#22948](https://github.com/sgl-project/sglang/pull/22948).

Additional open radar: [#19585](https://github.com/sgl-project/sglang/pull/19585), [#19781](https://github.com/sgl-project/sglang/pull/19781), [#19956](https://github.com/sgl-project/sglang/pull/19956), [#19974](https://github.com/sgl-project/sglang/pull/19974), [#20029](https://github.com/sgl-project/sglang/pull/20029), [#20255](https://github.com/sgl-project/sglang/pull/20255), [#20370](https://github.com/sgl-project/sglang/pull/20370), [#20448](https://github.com/sgl-project/sglang/pull/20448), [#20667](https://github.com/sgl-project/sglang/pull/20667), [#20789](https://github.com/sgl-project/sglang/pull/20789), [#20918](https://github.com/sgl-project/sglang/pull/20918), [#21185](https://github.com/sgl-project/sglang/pull/21185), [#21217](https://github.com/sgl-project/sglang/pull/21217), [#21464](https://github.com/sgl-project/sglang/pull/21464), [#22027](https://github.com/sgl-project/sglang/pull/22027), [#22325](https://github.com/sgl-project/sglang/pull/22325), [#22502](https://github.com/sgl-project/sglang/pull/22502), [#22867](https://github.com/sgl-project/sglang/pull/22867), [#22885](https://github.com/sgl-project/sglang/pull/22885), [#23062](https://github.com/sgl-project/sglang/pull/23062), [#23468](https://github.com/sgl-project/sglang/pull/23468), [#23471](https://github.com/sgl-project/sglang/pull/23471).

External evidence: official SGLang Qwen3.5 docs cover hybrid GDN/full-attention, shared experts, DeepStack Vision/Conv3d, `--attention-backend triton`, `SGLANG_USE_AITER=1`, `--reasoning-parser qwen3`, and `--tool-call-parser qwen3_coder`; AMD's day-0 article confirms the ROCm kernel path for GDN, shared-expert MoE, and vision kernels.
