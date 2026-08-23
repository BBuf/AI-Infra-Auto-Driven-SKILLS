# TokenSpeed Model PR Optimization History

Current model families:

- `kimi`
- `qwen35`

## Current Watch / Landed Items

Refresh: `2026-08-23`. Source head:
`lightseekorg/tokenspeed@2706143a8669d50a8f56466b9d340b86922b8f2d`.

| PR | Model / area | Status | Current signal | Why it matters |
| --- | --- | --- | --- | --- |
| [#1111](https://github.com/lightseekorg/tokenspeed/pull/1111) | Qwen3.8-27B | open 2026-08-15 | Hopper cookbook | Proposed TokenSpeed recipe. Not an audited history card and not a four-framework cookbook enablement. |
| [#1208](https://github.com/lightseekorg/tokenspeed/pull/1208) | scheduler | merged 2026-08-22 | TokenSpeed Scheduler 0.1.9 | Runtime/scheduler bump. Record the SHA; this is not a model-card rewrite. |
| [#1200](https://github.com/lightseekorg/tokenspeed/pull/1200) | Kimi K3 | merged 2026-08-22 | unrouted K3 decode projections | Performance path on K3 decode. Do not fold it into the audited Kimi cards until the diff is read. |
| [#1199](https://github.com/lightseekorg/tokenspeed/pull/1199) | MLA FP8 KV | merged 2026-08-22 | packing fallback | Correctness fallback for MLA FP8 KV packing. |
| [#821](https://github.com/lightseekorg/tokenspeed/pull/821) | Kimi K3 | merged | FlatKV KDA/MLA deployment contract | Documents NVIDIA B300 and AMD recipes, checkpoint-layout constraints, sidecar requirements, and validation caveats without claiming a generic benchmark result. |
| [#780](https://github.com/lightseekorg/tokenspeed/pull/780) | Qwen3.5 / multi-node | merged | topology-safe collectives and staging | Cross-node groups now select NCCL and overlapped Mamba inputs use per-step pinned buffers. |
| [#797](https://github.com/lightseekorg/tokenspeed/pull/797) | Kimi / DFlash | merged | incremental capture | Wires Kimi hidden-state capture into incremental DFlash projection. |
| [#795](https://github.com/lightseekorg/tokenspeed/pull/795) | Kimi-K2.7 | merged | EAGLE3.1 | Adds checkpoint-driven FC normalization and output semantics for the MLA speculator. |
| [#766](https://github.com/lightseekorg/tokenspeed/pull/766) | Qwen3.5 / FP8 | merged | mixed GDN projection loading | Fixes garbled output for checkpoints with FP8 qkv/z and BF16 b/a projections. |
| [#510](https://github.com/lightseekorg/tokenspeed/pull/510) | Qwen3.5 / DFlash | merged | native optimized DFlash | Adds fused KV materialization, incremental projection, FA4, and FP8 draft-cache support. |
| [#596](https://github.com/lightseekorg/tokenspeed/pull/596) | Kimi / EAGLE3 | merged | mixed-step hang fix | Aligns draft collective sizing across DP ranks and guards zero-token lm-head launches. |
| [#534](https://github.com/lightseekorg/tokenspeed/pull/534) | MXFP4 / MoE | merged | gathered activation-scale fix | Affects MXFP4 MoE correctness and performance interpretation. |
| [#528](https://github.com/lightseekorg/tokenspeed/pull/528) | GLM-5.2 / AMD | merged | initial support | Adds the first GLM-5.2 AMD path; it is outside the two manually audited model pages. |
| [#456](https://github.com/lightseekorg/tokenspeed/pull/456) | Qwen3.5 VLM | merged | packed QKV rotary layout | Optimizes Qwen vision FA4 rotary/QKV path and changes VLM trace shape. |
| [#354](https://github.com/lightseekorg/tokenspeed/pull/354) | Qwen3.5 + Kimi VLM | merged | generalized multimodal runtime | Adds shared video/multimodal plumbing used by model-specific VLM paths. |
| [#198](https://github.com/lightseekorg/tokenspeed/pull/198) | Qwen3.5 | merged | gated activation fusion | Fuses sigmoid/mul and removes a reshape copy in Qwen3.5 attention output. |
| [#196](https://github.com/lightseekorg/tokenspeed/pull/196) | Qwen3.5 | merged | fused q/k GemmaRMSNorm | Collapses two norm launches in Qwen3.5 attention prep. |
| [#477](https://github.com/lightseekorg/tokenspeed/pull/477) | Kimi VLM | merged | Kimi Vision FA4 QKV + RoPE | Kimi-side counterpart to packed vision QKV rotary work. |
| [#454](https://github.com/lightseekorg/tokenspeed/pull/454) | Kimi K2.5 | merged | AMD MXFP4 serving | Adds MXFP4 layer/backend path and validation for Kimi serving. |
| [#126](https://github.com/lightseekorg/tokenspeed/pull/126) | Kimi K2.5 | merged | fused lm_head GEMM | Adds Kimi-gated persistent lm_head GEMM. |

Read the per-model files for timelines and diff audit cards.
