# TokenSpeed Model PR Optimization History

Current model families:

- `kimi`
- `qwen35`

## Current Watch / Landed Items

Refresh: `2026-06-27`. Source head:
`lightseekorg/tokenspeed@d0a7faddb5ec0d4c6d037c4c3e6a781d2c5164a8`.

| PR | Model / area | Status | Current signal | Why it matters |
| --- | --- | --- | --- | --- |
| [#520](https://github.com/lightseekorg/tokenspeed/pull/520) | Qwen3.5 / Hopper | open | Hopper support | TokenSpeed Qwen3.5 comparisons should record whether the target image includes this non-Blackwell path. |
| [#510](https://github.com/lightseekorg/tokenspeed/pull/510) | Qwen3.5 / DFlash | open | DFlash support | Speculative or DFlash-enabled Qwen3.5 baselines may change when this branch lands. |
| [#534](https://github.com/lightseekorg/tokenspeed/pull/534) | MXFP4 / MoE | open | gathered activation-scale fix | Affects MXFP4 MoE correctness and performance interpretation. |
| [#528](https://github.com/lightseekorg/tokenspeed/pull/528) | GLM-5.2 / AMD | open | initial support | GLM-5.2 TokenSpeed coverage is still in flight. |
| [#382](https://github.com/lightseekorg/tokenspeed/pull/382) | GDN prefill | open | fused causal-conv1d and QKV split | Relevant for GDN/linear-attention profiler traces and overlap opportunities. |
| [#456](https://github.com/lightseekorg/tokenspeed/pull/456) | Qwen3.5 VLM | merged | packed QKV rotary layout | Optimizes Qwen vision FA4 rotary/QKV path and changes VLM trace shape. |
| [#354](https://github.com/lightseekorg/tokenspeed/pull/354) | Qwen3.5 + Kimi VLM | merged | generalized multimodal runtime | Adds shared video/multimodal plumbing used by model-specific VLM paths. |
| [#198](https://github.com/lightseekorg/tokenspeed/pull/198) | Qwen3.5 | merged | gated activation fusion | Fuses sigmoid/mul and removes a reshape copy in Qwen3.5 attention output. |
| [#196](https://github.com/lightseekorg/tokenspeed/pull/196) | Qwen3.5 | merged | fused q/k GemmaRMSNorm | Collapses two norm launches in Qwen3.5 attention prep. |
| [#477](https://github.com/lightseekorg/tokenspeed/pull/477) | Kimi VLM | merged | Kimi Vision FA4 QKV + RoPE | Kimi-side counterpart to packed vision QKV rotary work. |
| [#454](https://github.com/lightseekorg/tokenspeed/pull/454) | Kimi K2.5 | merged | AMD MXFP4 serving | Adds MXFP4 layer/backend path and validation for Kimi serving. |
| [#126](https://github.com/lightseekorg/tokenspeed/pull/126) | Kimi K2.5 | merged | fused lm_head GEMM | Adds Kimi-gated persistent lm_head GEMM. |

Read the per-model files for timelines and diff audit cards.
