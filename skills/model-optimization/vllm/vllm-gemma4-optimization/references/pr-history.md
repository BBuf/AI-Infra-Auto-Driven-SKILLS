# vLLM Gemma 4 PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Scope: Gemma 4 text, MoE, multimodal, reasoning, tool use, and quantized MoE serving.

## Landed PRs

### PR #38826 - Implement Google Gemma 4 architecture support

- Link: https://github.com/vllm-project/vllm/pull/38826
- Why it mattered: Initial Gemma 4 text/MoE/multimodal landing.
- Runtime path: vllm/vllm/model_executor/models/gemma4.py, vllm/vllm/model_executor/models/gemma4_mm.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #38879 - Enable Fast Prefill Optimization

- Link: https://github.com/vllm-project/vllm/pull/38879
- Why it mattered: Added YOCO KV-sharing based fast prefill for Gemma4.
- Runtime path: vllm/vllm/model_executor/models/gemma4.py, vllm/vllm/model_executor/models/gemma4_mm.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #39045 - Support quantized MoE

- Link: https://github.com/vllm-project/vllm/pull/39045
- Why it mattered: Extended Gemma4 to quantized MoE checkpoints.
- Runtime path: vllm/vllm/model_executor/models/gemma4.py, vllm/vllm/model_executor/models/gemma4_mm.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #38844 - Enable Gemma4ForCausalLM to load LoRA adapters correctly

- Link: https://github.com/vllm-project/vllm/pull/38844
- Why it mattered: Fixed adapter naming/load behavior.
- Runtime path: vllm/vllm/model_executor/models/gemma4.py, vllm/vllm/model_executor/models/gemma4_mm.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #39450 - Add Gemma4 Eagle3 support

- Link: https://github.com/vllm-project/vllm/pull/39450
- Why it mattered: Enabled speculative decode for Gemma4.
- Runtime path: vllm/vllm/model_executor/models/gemma4.py, vllm/vllm/model_executor/models/gemma4_mm.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.
