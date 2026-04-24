# SGLang Mixtral Quark INT4-FP8 MoE Support and Optimization Timeline

Evidence snapshot: SGLang `origin/main` `bca3dd958` on `2026-04-24`.

Scope: `mistralai/Mixtral-8x7B-Instruct-v0.1` with AMD-only `quark_int4fp8_moe` online MoE quantization.

## Summary

The Mixtral path validates SGLang's `quark_int4fp8_moe` quantization method: high-precision MoE weights are quantized online to packed INT4 and executed with FP8-style MoE math on ROCm. Current CI coverage is `test/registered/quant/test_int4fp8_moe.py`.

## Diff-Reviewed PR Cards

### #7392 - Add quark_int4fp8_moe online quantization on ROCm

- Link: https://github.com/sgl-project/sglang/pull/7392
- State: merged at `2026-01-14T09:44:41Z`
- Diff coverage: `4055` lines, `12` files.
- Adds the quantization config, INT4/FP8 utilities, server flag registration, docs, and the original Mixtral GSM8K regression.

### #17116 - Migrate AMD int4fp8 MoE test to registered CI

- Link: https://github.com/sgl-project/sglang/pull/17116
- State: merged at `2026-01-19T16:07:39Z`
- Diff coverage: `902` lines, `19` files.
- Moves the test into registered AMD CI and makes suite registration part of the support surface.

### #23455 - Restore int4fp8 MoE executable test

- Link: https://github.com/sgl-project/sglang/pull/23455
- State: merged at `2026-04-23T05:28:21Z`
- Diff coverage: `291` lines, `2` files.
- Restores `test/registered/quant/test_int4fp8_moe.py` with a direct `unittest.main()` entry and a GSM8K threshold above `0.56`.

Full PR dossier: `skills/model-optimization/sglang/sglang-mixtral-quark-int4fp8-moe-optimization/references/pr-history.md`.
