# vLLM Step3.5 / Step3-VL PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Scope: Step3.5-Flash and Step3-VL serving, NVFP4, tool/reasoning parser, and HF-style processor evolution.

## Landed PRs

### PR #33755 - Enable Step3p5ForCausalLM testing

- Link: https://github.com/vllm-project/vllm/pull/33755
- Why it mattered: Stabilized the core Step3.5 text runtime.
- Runtime path: vllm/vllm/model_executor/models/step3p5.py, vllm/vllm/model_executor/models/step3p5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #34478 - Add NVFP4 quantization support for Step3.5-Flash

- Link: https://github.com/vllm-project/vllm/pull/34478
- Why it mattered: Opened the practical quantized deployment path.
- Runtime path: vllm/vllm/model_executor/models/step3p5.py, vllm/vllm/model_executor/models/step3p5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #37579 - Refactor Step3-VL processor to HF style

- Link: https://github.com/vllm-project/vllm/pull/37579
- Why it mattered: Modernized the Step3-VL processor contract.
- Runtime path: vllm/vllm/model_executor/models/step3p5.py, vllm/vllm/model_executor/models/step3p5_mtp.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.
