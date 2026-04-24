# vLLM Hunyuan 3 Preview PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: partially supported or only adjacent architectures landed on current mainline
- Scope: Adjacent Hunyuan dense / OCR / VL support in vLLM relevant to Hunyuan 3 Preview planning, without a dedicated `Hunyuan3Preview` mainline alias yet.

## Landed PRs

### PR #21368 - Add Hunyuan V1 Dense Model support

- Link: https://github.com/vllm-project/vllm/pull/21368
- Why it mattered: Brought the dense Hunyuan line into vLLM mainline.
- Runtime path: vllm/vllm/model_executor/models/hunyuan_v1.py, vllm/vllm/model_executor/models/hunyuan_vision.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #29327 - Add HunyuanOCR support

- Link: https://github.com/vllm-project/vllm/pull/29327
- Why it mattered: Extended the family to OCR workloads instead of text-only generation.
- Runtime path: vllm/vllm/model_executor/models/hunyuan_v1.py, vllm/vllm/model_executor/models/hunyuan_vision.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #33035 - Eagle3 support for HunyuanVL & Hunyuan

- Link: https://github.com/vllm-project/vllm/pull/33035
- Why it mattered: Added speculative decoding support on top of the Hunyuan family.
- Runtime path: vllm/vllm/model_executor/models/hunyuan_v1.py, vllm/vllm/model_executor/models/hunyuan_vision.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

## Open PR Radar

- No specific open PR is pinned here; rerun GitHub PR search before asserting new support.
