# TensorRT-LLM Model PR Optimization History

Current model families:

- `kimi`
- `qwen35`

## Current Watch / Landed Items

Refresh: `2026-06-26`. Source head:
`NVIDIA/TensorRT-LLM@0722c5f47d2cae69ac1a237da51e550dd214532c`.

| PR | Model / area | Status | Current signal | Why it matters |
| --- | --- | --- | --- | --- |
| [#11685](https://github.com/NVIDIA/TensorRT-LLM/pull/11685) | KV cache runtime | merged | evict empty blocks first | Affects cache pressure and request residency under serving load; stale TensorRT-LLM images can mislead long-context or prefix-heavy rows. |
| [#15546](https://github.com/NVIDIA/TensorRT-LLM/pull/15546) | PyTorch executor KV cache | merged | fresh host buffer for KV block offsets | Affects race/overlap risk around KV block offset staging in benchmark and profiler traces. |
| [#15543](https://github.com/NVIDIA/TensorRT-LLM/pull/15543) | Qwen3.5 | merged | EPLB support | Changes Qwen3.5 MoE load-balancing behavior and benchmark fairness knobs. |
| [#15185](https://github.com/NVIDIA/TensorRT-LLM/pull/15185) | Qwen3.5 AutoDeploy | merged | sharding and lm_head sharding | Affects AutoDeploy/PyTorch backend memory and parallelism comparisons. |
| [#15067](https://github.com/NVIDIA/TensorRT-LLM/pull/15067) | Qwen3.5 | merged | FP8 checkpoint loading | Relevant when comparing dense/MoE Qwen3.5 checkpoints across frameworks. |
| [#15233](https://github.com/NVIDIA/TensorRT-LLM/pull/15233) | Kimi K2.5 | merged | rejection-sampling embedding mask | Affects speculative decoding / guided decoding comparisons. |
| [#15180](https://github.com/NVIDIA/TensorRT-LLM/pull/15180) | Kimi K2.5 | merged | guided decoding methods | Relevant when Kimi agentic/tool traces use guided decoding. |
| [#12788](https://github.com/NVIDIA/TensorRT-LLM/pull/12788) | Kimi K2.5 VLM | merged | multimodal vision support | Establishes TensorRT-LLM Kimi-K2.5 multimodal path and tests. |

Read the per-model files for timelines and diff audit cards.
