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

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# SGLANG Mixtral Quark INT4-FP8 MoE PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-06-20 | [#7392](https://github.com/sgl-project/sglang/pull/7392) | merged | [AMD][Quantization] Add `int4fp8_moe` online quantization on ROCm | MoE/router, quantization, tests/benchmarks, docs/config | `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`, `python/sglang/srt/layers/int4fp8_utils.py`, `test/srt/test_int4fp8_moe.py` |
| 2026-01-15 | [#17116](https://github.com/sgl-project/sglang/pull/17116) | merged | [AMD CI] Migrate and Add More Testcases | attention/backend, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks | `.github/workflows/pr-test-amd.yml`, `test/registered/amd/test_deepseek_v3_mtp.py`, `test/registered/amd/test_deepseek_v3_basic.py` |
| 2026-04-22 | [#23455](https://github.com/sgl-project/sglang/pull/23455) | merged | [AMD] Restore test_zimage_turbo.py and test_int4fp8_moe.py with __main__ entry | MoE/router, quantization, multimodal/processor, tests/benchmarks | `test/registered/amd/test_zimage_turbo.py`, `test/registered/quant/test_int4fp8_moe.py` |

## Diff Cards

### PR #7392 - [AMD][Quantization] Add `int4fp8_moe` online quantization on ROCm

- Link: https://github.com/sgl-project/sglang/pull/7392
- Status/date: `merged`, created 2025-06-20, merged 2026-01-14; author `fxmarty-amd`.
- Diff scope read: `12` files, `+615/-15`; areas: MoE/router, quantization, tests/benchmarks, docs/config; keywords: fp8, quant, moe, config, cache, spec, triton, attention, awq, cuda.
- Code diff details:
  - `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py` added +443/-0 (443 lines); hunks: +import logging; symbols: tqdm_reset_no_print, QuarkInt4Fp8Config, for, __init__
  - `python/sglang/srt/layers/int4fp8_utils.py` added +73/-0 (73 lines); hunks: +"""; symbols: quantize_fp8_scale_tensorwise, quantize_int4_scale_columnwise, pack_int4_to_int32
  - `test/srt/test_int4fp8_moe.py` added +55/-0 (55 lines); hunks: +from types import SimpleNamespace; symbols: TestMixtralAccuracy, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/model_loader/weight_utils.py` modified +16/-14 (30 lines); hunks: ci_download_with_validation_and_retry,; def filter_files_not_needed_for_inference(hf_weights_files: List[str]) -> List[s; symbols: filter_files_not_needed_for_inference, np_cache_weights_iterator, np_cache_weights_iterator, safetensors_weights_iterator
  - `docs/advanced_features/quantization.md` modified +8/-0 (8 lines); hunks: python3 -m sglang.launch_server \; python3 -m sglang.launch_server \
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`, `python/sglang/srt/layers/int4fp8_utils.py`, `test/srt/test_int4fp8_moe.py`; keywords observed in patches: fp8, quant, moe, config, cache, spec. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/quark_int4fp8_moe.py`, `python/sglang/srt/layers/int4fp8_utils.py`, `test/srt/test_int4fp8_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17116 - [AMD CI] Migrate and Add More Testcases

- Link: https://github.com/sgl-project/sglang/pull/17116
- Status/date: `merged`, created 2026-01-15, merged 2026-01-19; author `bingxche`.
- Diff scope read: `19` files, `+310/-66`; areas: attention/backend, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks; keywords: test, cache, cuda, fp8, kv, moe, attention, config, lora, topk.
- Code diff details:
  - `.github/workflows/pr-test-amd.yml` modified +81/-47 (128 lines); hunks: jobs:; jobs:
  - `test/registered/amd/test_deepseek_v3_mtp.py` added +116/-0 (116 lines); hunks: +import unittest; symbols: TestDeepseekV3MTP, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/amd/test_deepseek_v3_basic.py` added +84/-0 (84 lines); hunks: +import unittest; symbols: TestDeepseekV3Basic, setUpClass, tearDownClass, test_a_gsm8k
  - `test/srt/run_suite.py` modified +0/-8 (8 lines); hunks: # TestFile("lora/test_lora_backend.py", 99), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/13107; # TestFile("test_vision_chunked_pre
  - `test/registered/core/test_deterministic.py` modified +5/-1 (6 lines); hunks: import unittest; def get_server_args(cls):; symbols: TestFlashinferDeterministic, get_server_args, TestFa3Deterministic
- Optimization/support interpretation: The concrete diff surface is `.github/workflows/pr-test-amd.yml`, `test/registered/amd/test_deepseek_v3_mtp.py`, `test/registered/amd/test_deepseek_v3_basic.py`; keywords observed in patches: test, cache, cuda, fp8, kv, moe. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `.github/workflows/pr-test-amd.yml`, `test/registered/amd/test_deepseek_v3_mtp.py`, `test/registered/amd/test_deepseek_v3_basic.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23455 - [AMD] Restore test_zimage_turbo.py and test_int4fp8_moe.py with __main__ entry

- Link: https://github.com/sgl-project/sglang/pull/23455
- Status/date: `merged`, created 2026-04-22, merged 2026-04-23; author `bingxche`.
- Diff scope read: `2` files, `+220/-0`; areas: MoE/router, quantization, multimodal/processor, tests/benchmarks; keywords: test, attention, config, fp8, moe, processor, quant, triton.
- Code diff details:
  - `test/registered/amd/test_zimage_turbo.py` added +156/-0 (156 lines); hunks: +"""AMD nightly test for Z-Image-Turbo diffusion model (text-to-image)."""; symbols: _save_image_and_write_summary, _compute_clip_score, TestZImageTurboAMD, teardown_class
  - `test/registered/quant/test_int4fp8_moe.py` added +64/-0 (64 lines); hunks: +from types import SimpleNamespace; symbols: TestMixtralAccuracy, setUpClass, tearDownClass, test_gsm8k
- Optimization/support interpretation: The concrete diff surface is `test/registered/amd/test_zimage_turbo.py`, `test/registered/quant/test_int4fp8_moe.py`; keywords observed in patches: test, attention, config, fp8, moe, processor. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/amd/test_zimage_turbo.py`, `test/registered/quant/test_int4fp8_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
