# TokenSpeed Model PR Optimization History

Current model families:

- `kimi`
- `qwen35`

## Current Watch / Landed Items

Refresh: `2026-06-26`. Source head:
`lightseekorg/tokenspeed@5aedf69d6b476baa65571011de6ea60fd5a238a8`.

| PR | Model / area | Status | Current signal | Why it matters |
| --- | --- | --- | --- | --- |
| [#456](https://github.com/lightseekorg/tokenspeed/pull/456) | Qwen3.5 VLM | merged | packed QKV rotary layout | Optimizes Qwen vision FA4 rotary/QKV path and changes VLM trace shape. |
| [#354](https://github.com/lightseekorg/tokenspeed/pull/354) | Qwen3.5 + Kimi VLM | merged | generalized multimodal runtime | Adds shared video/multimodal plumbing used by model-specific VLM paths. |
| [#198](https://github.com/lightseekorg/tokenspeed/pull/198) | Qwen3.5 | merged | gated activation fusion | Fuses sigmoid/mul and removes a reshape copy in Qwen3.5 attention output. |
| [#196](https://github.com/lightseekorg/tokenspeed/pull/196) | Qwen3.5 | merged | fused q/k GemmaRMSNorm | Collapses two norm launches in Qwen3.5 attention prep. |
| [#477](https://github.com/lightseekorg/tokenspeed/pull/477) | Kimi VLM | merged | Kimi Vision FA4 QKV + RoPE | Kimi-side counterpart to packed vision QKV rotary work. |
| [#454](https://github.com/lightseekorg/tokenspeed/pull/454) | Kimi K2.5 | merged | AMD MXFP4 serving | Adds MXFP4 layer/backend path and validation for Kimi serving. |
| [#126](https://github.com/lightseekorg/tokenspeed/pull/126) | Kimi K2.5 | merged | fused lm_head GEMM | Adds Kimi-gated persistent lm_head GEMM. |

Read the per-model files for timelines and diff audit cards.
