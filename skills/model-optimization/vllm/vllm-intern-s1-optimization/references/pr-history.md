# vLLM Intern-S1 PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Scope: Intern-S1 language and video-aware serving, processor integration, and tool/reasoning parser behavior.

## Landed PRs

### PR #21628 - Support Intern-S1

- Link: https://github.com/vllm-project/vllm/pull/21628
- Why it mattered: Initial Intern-S1 support in vLLM.
- Runtime path: vllm/vllm/model_executor/models/interns1.py, vllm/vllm/model_executor/models/interns1_pro.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #21671 - Add video support for Intern-S1

- Link: https://github.com/vllm-project/vllm/pull/21671
- Why it mattered: Extended the family beyond static images.
- Runtime path: vllm/vllm/model_executor/models/interns1.py, vllm/vllm/model_executor/models/interns1_pro.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #22417 - Fix wrong method name in Intern-S1 image processor

- Link: https://github.com/vllm-project/vllm/pull/22417
- Why it mattered: Patched a processor bug after bring-up.
- Runtime path: vllm/vllm/model_executor/models/interns1.py, vllm/vllm/model_executor/models/interns1_pro.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #33636 - Intern-S1-Pro

- Link: https://github.com/vllm-project/vllm/pull/33636
- Why it mattered: Added the Pro generation / alias path.
- Runtime path: vllm/vllm/model_executor/models/interns1.py, vllm/vllm/model_executor/models/interns1_pro.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.
