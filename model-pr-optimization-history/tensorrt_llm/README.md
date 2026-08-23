# TensorRT-LLM Model PR Optimization History

Current model families:

- `kimi`
- `qwen35`

## Current Watch / Landed Items

Refresh: `2026-08-23`. Source head:
`NVIDIA/TensorRT-LLM@da38c1d2e0dffd073b7dfb6d69e15ee7b45d84a9`.

2026-08-23 live main also contains recent Qwen3.8 / Kimi K3 merges that
are **not** folded into audited cards: Qwen3.5/3.8 wave-2 `#17700`,
Qwen3.8-27B FP8 VLM quant-config `#17786`, Kimi K3 MLA decode backend
`#17800`, and Kimi K3 NVFP4 MegaMoE SiTU `#17865`. Read those diffs
before treating them as TensorRT-LLM history evidence.

2026-08-23 open-PR watch (also not folded into audited cards): DeepSeek-V4
Hopper `#16940`, Kimi K3 frontend/tests/tool-call `#17822` / `#17921` /
`#17980`, MiniMax-M3 MSA/MXFP8 `#17236` / `#17238` / `#17318`. Read the
diffs before treating any of these as shipped TensorRT-LLM behavior.
The final one-commit increment is PR
[#16677](https://github.com/NVIDIA/TensorRT-LLM/pull/16677), which enables
Attention2D plus tensor parallelism for VisualGen/Wan and is intentionally not
promoted as Kimi or Qwen3.5 LLM evidence.

| PR | Model / area | Status | Current signal | Why it matters |
| --- | --- | --- | --- | --- |
| [#16805](https://github.com/NVIDIA/TensorRT-LLM/pull/16805) | disaggregated speculative runtime | merged | draft-token and sequence-length accounting | Adopts draft tokens from context-phase handoff and counts both first-generation and draft tokens on decode. |
| [#16763](https://github.com/NVIDIA/TensorRT-LLM/pull/16763) | PyTorch executor startup | merged | unified phase-1 CUDA graph cleanup | Avoids duplicate graph release while rebuilding the final KV cache after capacity estimation. |
| [#16469](https://github.com/NVIDIA/TensorRT-LLM/pull/16469) | Qwen3.5/3.6 attention | merged | fused QK norm + RoPE + gate | Collapses attention preprocessing and output gating launches; compare traces only after recording this fused path. |
| [#15194](https://github.com/NVIDIA/TensorRT-LLM/pull/15194) | Qwen3-Next / Qwen3.5 | merged | Gemma RMSNorm + AllReduce | Changes TP collective ownership and removes standalone norm work. |
| [#14848](https://github.com/NVIDIA/TensorRT-LLM/pull/14848) | Kimi K2.5 / NVFP4 | merged | RMSNorm + FP4 quant fusion | Adds the Blackwell fused normalization/quantization edge used by Kimi-style MLA. |
| [#15249](https://github.com/NVIDIA/TensorRT-LLM/pull/15249) | Qwen3.5-VL Dense | merged | dense multimodal support | Establishes the dense VLM model/config/weight-mapper and parity lane. |
| [#14599](https://github.com/NVIDIA/TensorRT-LLM/pull/14599) | Qwen3.5-VL MoE | merged | MoE VLM + MTP fixes | Establishes the MoE multimodal wrapper and speculative token plumbing. |
| [#15594](https://github.com/NVIDIA/TensorRT-LLM/pull/15594) | Qwen3.5 GDN | merged | piecewise CUDA graph capture fix | Captures the GDN path that older images left outside piecewise graphs. |
| [#11685](https://github.com/NVIDIA/TensorRT-LLM/pull/11685) | KV cache runtime | merged | evict empty blocks first | Affects cache pressure and request residency under serving load; stale TensorRT-LLM images can mislead long-context or prefix-heavy rows. |
| [#15546](https://github.com/NVIDIA/TensorRT-LLM/pull/15546) | PyTorch executor KV cache | merged | fresh host buffer for KV block offsets | Affects race/overlap risk around KV block offset staging in benchmark and profiler traces. |
| [#15543](https://github.com/NVIDIA/TensorRT-LLM/pull/15543) | Qwen3.5 | merged | EPLB support | Changes Qwen3.5 MoE load-balancing behavior and benchmark fairness knobs. |
| [#15185](https://github.com/NVIDIA/TensorRT-LLM/pull/15185) | Qwen3.5 AutoDeploy | merged | sharding and lm_head sharding | Affects AutoDeploy/PyTorch backend memory and parallelism comparisons. |
| [#15067](https://github.com/NVIDIA/TensorRT-LLM/pull/15067) | Qwen3.5 | merged | FP8 checkpoint loading | Relevant when comparing dense/MoE Qwen3.5 checkpoints across frameworks. |
| [#15233](https://github.com/NVIDIA/TensorRT-LLM/pull/15233) | Kimi K2.5 | merged | rejection-sampling embedding mask | Affects speculative decoding / guided decoding comparisons. |
| [#15180](https://github.com/NVIDIA/TensorRT-LLM/pull/15180) | Kimi K2.5 | merged | guided decoding methods | Relevant when Kimi agentic/tool traces use guided decoding. |
| [#12788](https://github.com/NVIDIA/TensorRT-LLM/pull/12788) | Kimi K2.5 VLM | merged | multimodal vision support | Establishes TensorRT-LLM Kimi-K2.5 multimodal path and tests. |

Read the per-model files for timelines and diff audit cards.
