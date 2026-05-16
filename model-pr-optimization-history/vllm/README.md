# vLLM Model PR Optimization History

Current model families:

- `deepseek-v3-r1`
- `deepseek-v31`
- `deepseek-v32`
- `deepseek-v4`
- `ernie45`
- `gemma4`
- `glm-vlm-ocr`
- `glm45`
- `glm46-glm47`
- `glm5-glm51`
- `gpt-oss`
- `hunyuan3-preview`
- `intern-s1`
- `internvl35`
- `kimi`
- `llama4`
- `mimo-v2-flash`
- `minimax`
- `mistral-small-4`
- `mixtral-quark-int4fp8-moe`
- `moss-vl`
- `nemotron-super`
- `qwen-vlm-omni-asr`
- `qwen3-coder`
- `qwen3-core`
- `qwen3-next`
- `qwen35`
- `qwen36`
- `step35`

## Current Watch / Landed Items

Refresh: `2026-05-15`. Keep watch rows close to the relevant model histories;
landed rows that are already mirrored in per-model docs remain here as
cross-model navigation hints.

| PR | Model / area | Status | Current signal | Why it matters |
| --- | --- | --- | --- | --- |
| [#41455](https://github.com/vllm-project/vllm/pull/41455) | ROCm attention | watch | WMMA paged prefill and split-K decode | New AMD attention kernel family for prefill/decode split traces. |
| [#41263](https://github.com/vllm-project/vllm/pull/41263) | DeepSeek-V4 | landed | fuse norm/router low latency | Mirrored in `vllm/deepseek-v4`; concrete DSV4 norm-router fusion precedent. |
| [#41428](https://github.com/vllm-project/vllm/pull/41428) | DeepSeek-V4 | landed | fused indexer Q quant | Mirrored in `vllm/deepseek-v4`; relevant to FP4 indexer-Q quant ladders. |
| [#41255](https://github.com/vllm-project/vllm/pull/41255) | DeepSeek-V4 | landed | Tile kernels and `head_compute_mix_kernel` | Mirrored in `vllm/deepseek-v4`; specialized MLA/head-compute kernel work. |
| [#41441](https://github.com/vllm-project/vllm/pull/41441) | DeepSeek-V4 | watch | all-reduce plus `mhc_post` fusion | Affects collective-plus-postprocess overlap reads. |
| [#37646](https://github.com/vllm-project/vllm/pull/37646) | ROCm TP | watch | AITER fused allreduce plus RMSNorm | AMD counterpart to FlashInfer allreduce fusion. |
| [#39748](https://github.com/vllm-project/vllm/pull/39748) | Qwen3 / Qwen3.5 GDN | watch | dual-stream input projection | Precedent for overlapping linear-attention projection branches. |
| [#41446](https://github.com/vllm-project/vllm/pull/41446) | GatedDeltaNet / FLA | watch | AMD prefill kernels | Relevant for Qwen3-Next-style linear-attention prefill bottlenecks. |
| [#41375](https://github.com/vllm-project/vllm/pull/41375) | sampler | watch | warm up `forward_native` sampler kernel | Avoids first-hit sampler artifacts in profiler and benchmark comparisons. |
| [#36823](https://github.com/vllm-project/vllm/pull/36823) | vLLM IR | watch | `fused_add_rms_norm` overload visibility | Can determine whether downstream norm-plus-quant compile fusions match. |
| [#41433](https://github.com/vllm-project/vllm/pull/41433) / [#41434](https://github.com/vllm-project/vllm/pull/41434) / [#41429](https://github.com/vllm-project/vllm/pull/41429) / [#40561](https://github.com/vllm-project/vllm/pull/40561) | runtime sync | watch | GPU/CPU sync removal and checks | Profiler gaps may be sync-removal work, not kernel fusion work. |
