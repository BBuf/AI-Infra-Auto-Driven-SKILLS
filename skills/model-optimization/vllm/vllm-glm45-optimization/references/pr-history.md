# vLLM GLM-4.5 / 4.5V PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Scope: GLM-4.5 text, GLM-4.5V, GLM-4.5-Air, shared MoE routing, and tool/reasoning parser behavior in vLLM.


## Landed PRs

### PR #22171 - Modify the organization of GLM series

- Link: https://github.com/vllm-project/vllm/pull/22171
- Why it mattered: Reworked the family layout so 4.5-era models reused a cleaner GLM structure.
- Runtime path: vllm/vllm/model_executor/models/glm4.py, vllm/vllm/model_executor/models/glm4_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #22460 - not tie_word_embeddings for glm-4.5 and glm-4.5v

- Link: https://github.com/vllm-project/vllm/pull/22460
- Why it mattered: Aligned the loader with the real 4.5 checkpoint contract instead of forcing tied embeddings.
- Runtime path: vllm/vllm/model_executor/models/glm4.py, vllm/vllm/model_executor/models/glm4_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #22832 - Modify the gate implementation of glm4_moe

- Link: https://github.com/vllm-project/vllm/pull/22832
- Why it mattered: Changed the GLM4.5 MoE gating path used by text and VL variants.
- Runtime path: vllm/vllm/model_executor/models/glm4.py, vllm/vllm/model_executor/models/glm4_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #23695 - Add triton fused moe config for GLM-4.5-Air-FP8 on B200

- Link: https://github.com/vllm-project/vllm/pull/23695
- Why it mattered: Added a production kernel-tuning lane for the 4.5 Air FP8 deployment path.
- Runtime path: vllm/vllm/model_executor/models/glm4.py, vllm/vllm/model_executor/models/glm4_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #24589 - Add documentation for GLM-4.5 series tool-calling and reasoning parser

- Link: https://github.com/vllm-project/vllm/pull/24589
- Why it mattered: Codified the parser choices needed to serve 4.5 reasoning / tool checkpoints correctly.
- Runtime path: vllm/vllm/model_executor/models/glm4.py, vllm/vllm/model_executor/models/glm4_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

## Open PR Radar

- No specific open PR is pinned here; rerun GitHub PR search before asserting new support.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# VLLM GLM-4.5 PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-08-04 | [#22171](https://github.com/vllm-project/vllm/pull/22171) | merged | [Misc] Modify the organization of GLM series | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `docs/models/supported_models.md`, `tests/models/registry.py`, `tests/models/multimodal/generation/test_common.py` |
| 2025-08-07 | [#22460](https://github.com/vllm-project/vllm/pull/22460) | merged | not tie_word_embeddings for glm-4.5 and glm-4.5v | model wrapper, MoE/router, scheduler/runtime | `vllm/model_executor/models/glm4_moe.py` |
| 2025-08-13 | [#22832](https://github.com/vllm-project/vllm/pull/22832) | merged | [Model] Modify the gate implementation of glm4_moe | model wrapper, MoE/router, scheduler/runtime, docs/config | `vllm/model_executor/models/glm4_moe.py`, `docs/models/supported_models.md` |
| 2025-08-26 | [#23695](https://github.com/vllm-project/vllm/pull/23695) | merged | feat: add triton fused moe config for GLM-4.5-Air-FP8 on B200 | MoE/router, quantization, scheduler/runtime, docs/config | `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` |
| 2025-09-10 | [#24589](https://github.com/vllm-project/vllm/pull/24589) | merged | [Doc] Add documentation for GLM-4.5 series models: tool-calling and reasoning parser | docs/config | `docs/features/tool_calling.md`, `docs/features/reasoning_outputs.md` |

## Diff Cards

### PR #22171 - [Misc] Modify the organization of GLM series

- Link: https://github.com/vllm-project/vllm/pull/22171
- Status/date: `merged`, created 2025-08-04, merged 2025-08-04; author `jeejeelee`.
- Diff scope read: `16` files, `+31/-31`; areas: model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: test, moe, config, doc, fp8, lora, vision.
- Code diff details:
  - `docs/models/supported_models.md` modified +5/-5 (10 lines); hunks: th {; th {
  - `tests/models/registry.py` modified +5/-5 (10 lines); hunks: def check_available_online(; def check_available_online(; symbols: check_available_online, check_available_online, check_available_online
  - `tests/models/multimodal/generation/test_common.py` modified +3/-3 (6 lines); hunks: num_logprobs=10,; marks=[large_gpu_mark(min_gb=32)],
  - `vllm/model_executor/models/chatglm.py` modified +3/-3 (6 lines); hunks: # SPDX-License-Identifier: Apache-2.0; def __init__(; symbols: __init__
  - `examples/offline_inference/vision_language.py` modified +2/-2 (4 lines); hunks: def run_gemma3(questions: list[str], modality: str) -> ModelRequestData:; def run_glm4v(questions: list[str], modality: str) -> ModelRequestData:; symbols: run_gemma3, run_glm4v, run_glm4v, run_glm4_1v
- Optimization/support interpretation: The concrete diff surface is `docs/models/supported_models.md`, `tests/models/registry.py`, `tests/models/multimodal/generation/test_common.py`; keywords observed in patches: test, moe, config, doc, fp8, lora. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/models/supported_models.md`, `tests/models/registry.py`, `tests/models/multimodal/generation/test_common.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22460 - not tie_word_embeddings for glm-4.5 and glm-4.5v

- Link: https://github.com/vllm-project/vllm/pull/22460
- Status/date: `merged`, created 2025-08-07, merged 2025-08-08; author `zRzRzRzRzRzRzR`.
- Diff scope read: `1` files, `+0/-2`; areas: model wrapper, MoE/router, scheduler/runtime; keywords: config, moe, processor, quant.
- Code diff details:
  - `vllm/model_executor/models/glm4_moe.py` modified +0/-2 (2 lines); hunks: def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/glm4_moe.py`; keywords observed in patches: config, moe, processor, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/glm4_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22832 - [Model] Modify the gate implementation of glm4_moe

- Link: https://github.com/vllm-project/vllm/pull/22832
- Status/date: `merged`, created 2025-08-13, merged 2025-08-14; author `jeejeelee`.
- Diff scope read: `2` files, `+11/-11`; areas: model wrapper, MoE/router, scheduler/runtime, docs/config; keywords: moe, config, doc, expert, kv, processor, quant, router.
- Code diff details:
  - `vllm/model_executor/models/glm4_moe.py` modified +10/-10 (20 lines); hunks: from vllm.model_executor.layers.layernorm import RMSNorm; def __init__(; symbols: __init__, forward
  - `docs/models/supported_models.md` modified +1/-1 (2 lines); hunks: These models primarily accept the [`LLM.generate`](./generative_models.md#llmgen
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/glm4_moe.py`, `docs/models/supported_models.md`; keywords observed in patches: moe, config, doc, expert, kv, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/glm4_moe.py`, `docs/models/supported_models.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23695 - feat: add triton fused moe config for GLM-4.5-Air-FP8 on B200

- Link: https://github.com/vllm-project/vllm/pull/23695
- Status/date: `merged`, created 2025-08-26, merged 2025-08-27; author `zixuanzhang226`.
- Diff scope read: `1` files, `+146/-0`; areas: MoE/router, quantization, scheduler/runtime, docs/config; keywords: config, fp8, moe.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunks: +{
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`; keywords observed in patches: config, fp8, moe. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #24589 - [Doc] Add documentation for GLM-4.5 series models: tool-calling and reasoning parser

- Link: https://github.com/vllm-project/vllm/pull/24589
- Status/date: `merged`, created 2025-09-10, merged 2025-09-10; author `WangErXiao`.
- Diff scope read: `2` files, `+10/-0`; areas: docs/config; keywords: doc.
- Code diff details:
  - `docs/features/tool_calling.md` modified +9/-0 (9 lines); hunks: Flags:
  - `docs/features/reasoning_outputs.md` modified +1/-0 (1 lines); hunks: vLLM currently supports the following reasoning models:
- Optimization/support interpretation: The concrete diff surface is `docs/features/tool_calling.md`, `docs/features/reasoning_outputs.md`; keywords observed in patches: doc. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/features/tool_calling.md`, `docs/features/reasoning_outputs.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
