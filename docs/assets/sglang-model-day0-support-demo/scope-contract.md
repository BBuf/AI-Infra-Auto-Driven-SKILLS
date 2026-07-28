# Day-0 Scope Contract

## Release Cut

- Model: `Aurora-Hybrid-70B-VL`
- Checkpoint revision: `1111111111111111111111111111111111111111`
- SGLang revision: `2222222222222222222222222222222222222222`
- Primary platform: NVIDIA B200, eight GPUs
- Target release date: 2026-08-15
- Owner: public model-support working group

Day-0 covers one immutable fictional checkpoint and a correctness-first eager
fallback. Optimized kernels are optional unless the model cannot fit.

## Required Capabilities

| Lane | Required value | Success criterion |
| --- | --- | --- |
| Checkpoint | BF16 text-plus-vision | Complete weight audit and deterministic short generation |
| API | Reasoning and one tool call | Streaming/non-streaming fields match |
| Topology | TP8 unified and TP4+TP4 PD | Correctness, state transfer, and liveness pass |
| Speculative | Remote-loaded MTP draft | Target parity for accepted lengths zero through four |
| Multimodal | One RGB image | Processor and server embeddings agree |

## Out of Scope

- AMD and NPU execution
- Quantized MoE kernels
- Multi-image and video requests
