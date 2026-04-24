# SGLang Mixtral Quark INT4-FP8 MoE PR History

Evidence snapshot: SGLang `origin/main` `bca3dd958` on `2026-04-24`.

Scope: `mistralai/Mixtral-8x7B-Instruct-v0.1` validation for AMD-only `quark_int4fp8_moe` online MoE quantization.

## Current-Main Code Surfaces

- `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`
- `python/sglang/srt/layers/int4fp8_utils.py`
- `python/sglang/srt/layers/quantization/__init__.py`
- `python/sglang/srt/server_args.py`
- `python/sglang/srt/configs/model_config.py`
- `test/registered/quant/test_int4fp8_moe.py`
- `docs_new/docs/advanced_features/quantization.mdx`

## PR #7392 - Add quark_int4fp8_moe online quantization on ROCm

- Link: https://github.com/sgl-project/sglang/pull/7392
- State: merged at `2026-01-14T09:44:41Z`
- Diff coverage: full diff fetched, `4055` lines, `12` files.
- Motivation: support MoE checkpoints loaded in high precision, quantize expert weights online to INT4, and execute MoE with FP8 math on AMD.
- Key implementation: register `quark_int4fp8_moe`, add `QuarkInt4Fp8Config`, add INT4 packing/scale utilities, quantize experts during weight loading, and add a Mixtral GSM8K test.

```python
class QuarkInt4Fp8Config(QuantizationConfig):
    def get_name(self) -> str:
        return "quark_int4fp8_moe"
```

```python
_, fp8_scale = quantize_fp8_scale_tensorwise(loaded_weight)
int4_w, int4_scale = quantize_int4_scale_columnwise(loaded_weight)
int4_w = pack_int4_to_int32(int4_w)
```

- Validation implications: test on ROCm only, with real MoE weights and GSM8K accuracy rather than unit-only checks.

## PR #17116 - Migrate AMD int4fp8 MoE test to registered CI

- Link: https://github.com/sgl-project/sglang/pull/17116
- State: merged at `2026-01-19T16:07:39Z`
- Diff coverage: full diff fetched, `902` lines, `19` files.
- Motivation: reorganize AMD CI and migrate quantization tests from legacy locations to `test/registered`.
- Key implementation: move `test_int4fp8_moe.py` into registered quant tests and wire it into AMD stage-B coverage.
- Validation implications: suite names and CI registration are part of the contract; stale suite names can silently drop coverage.

## PR #23455 - Restore int4fp8 MoE direct-file test execution

- Link: https://github.com/sgl-project/sglang/pull/23455
- State: merged at `2026-04-23T05:28:21Z`
- Diff coverage: full diff fetched, `291` lines, `2` files.
- Motivation: #23305 removed tests that lacked executable `__main__` entries; the Mixtral int4fp8 MoE regression needed to be restored with a direct unittest entry.
- Key implementation: re-add `test/registered/quant/test_int4fp8_moe.py`, register it to AMD stage-B, launch Mixtral with `--quantization quark_int4fp8_moe`, and assert GSM8K score above `0.56`.

```python
other_args = [
    "--tp", "2",
    "--quantization", "quark_int4fp8_moe",
    "--attention-backend", "triton",
]
```

```python
if __name__ == "__main__":
    import unittest

    unittest.main()
```

- Validation implications: run the test by path on AMD, not just through pytest discovery.
