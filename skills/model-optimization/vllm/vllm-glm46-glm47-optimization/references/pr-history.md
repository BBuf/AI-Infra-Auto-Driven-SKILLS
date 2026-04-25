# vLLM GLM-4.6 / 4.7 PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: supported on current mainline
- Scope: GLM-4.6, GLM-4.6V, GLM-4.7, GLM-4.7-Flash, GLM-Lite, and the parser / quant / fused-MoE deltas after the 4.5 generation.


## Landed PRs

### PR #26818 - Add MoE tunings for GLM 4.6-FP8 and GLM 4.5 Air on B200

- Link: https://github.com/vllm-project/vllm/pull/26818
- Why it mattered: Added fused-MoE tuning configs for the new Blackwell deployment lane.
- Runtime path: vllm/vllm/model_executor/models/glm4.py, vllm/vllm/model_executor/models/glm4_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #30210 - Fix glm46 awq marlin moe compatibility

- Link: https://github.com/vllm-project/vllm/pull/30210
- Why it mattered: Closed an incompatibility between GLM-4.6 AWQ checkpoints and Marlin MoE assumptions.
- Runtime path: vllm/vllm/model_executor/models/glm4.py, vllm/vllm/model_executor/models/glm4_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #30876 - GLM-4.7 Tool Parser and Doc Update

- Link: https://github.com/vllm-project/vllm/pull/30876
- Why it mattered: Brought parser behavior and docs up to date for 4.7 / 4.7-Flash.
- Runtime path: vllm/vllm/model_executor/models/glm4.py, vllm/vllm/model_executor/models/glm4_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #31386 - GLM Model support for GLM-Lite

- Link: https://github.com/vllm-project/vllm/pull/31386
- Why it mattered: Extended the same runtime family to the Lite checkpoint line.
- Runtime path: vllm/vllm/model_executor/models/glm4.py, vllm/vllm/model_executor/models/glm4_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #37386 - Improve tool call parsing and content normalization for glm47

- Link: https://github.com/vllm-project/vllm/pull/37386
- Why it mattered: Fixed concrete parsing errors that surfaced in newer GLM-4.7 outputs.
- Runtime path: vllm/vllm/model_executor/models/glm4.py, vllm/vllm/model_executor/models/glm4_moe.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

## Open PR Radar

- No specific open PR is pinned here; rerun GitHub PR search before asserting new support.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# VLLM GLM-4.6 / GLM-4.7 PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-10-14 | [#26818](https://github.com/vllm-project/vllm/pull/26818) | merged | [Kernel][MoE] Add MoE tunings for GLM 4.6-FP8 and GLM 4.5 Air on NVidia B200 | MoE/router, quantization, scheduler/runtime, docs/config | `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json` |
| 2025-12-07 | [#30210](https://github.com/vllm-project/vllm/pull/30210) | merged | [Bugfix]: Fix glm46 awq marlin moe wna16 compatibility | MoE/router, quantization, scheduler/runtime | `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/quantization/moe_wna16.py` |
| 2025-12-17 | [#30876](https://github.com/vllm-project/vllm/pull/30876) | merged | GLM-4.7 Tool Parser and Doc Update | model wrapper, MoE/router, scheduler/runtime, docs/config | `vllm/tool_parsers/glm47_moe_tool_parser.py`, `docs/features/tool_calling.md`, `vllm/tool_parsers/__init__.py` |
| 2025-12-26 | [#31386](https://github.com/vllm-project/vllm/pull/31386) | merged | [GLM-4.7] GLM Model support for GLM-Lite | model wrapper, MoE/router, kernel, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/glm4_moe_lite.py`, `vllm/model_executor/models/glm4_moe_lite_mtp.py`, `vllm/config/speculative.py` |
| 2026-03-18 | [#37386](https://github.com/vllm-project/vllm/pull/37386) | merged | fix(glm47): improve tool call parsing and content normalization | MoE/router, tests/benchmarks | `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py` |

## Diff Cards

### PR #26818 - [Kernel][MoE] Add MoE tunings for GLM 4.6-FP8 and GLM 4.5 Air on NVidia B200

- Link: https://github.com/vllm-project/vllm/pull/26818
- Status/date: `merged`, created 2025-10-14, merged 2025-10-14; author `zklapow`.
- Diff scope read: `3` files, `+441/-0`; areas: MoE/router, quantization, scheduler/runtime, docs/config; keywords: config, moe, triton, fp8.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json` added +147/-0 (147 lines); hunks: +{
  - `vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json` added +147/-0 (147 lines); hunks: +{
  - `vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json` added +147/-0 (147 lines); hunks: +{
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json`; keywords observed in patches: config, moe, triton, fp8. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/layers/fused_moe/configs/E=32,N=1408,device_name=NVIDIA_B200.json`, `vllm/model_executor/layers/fused_moe/configs/E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json`, `vllm/model_executor/layers/fused_moe/configs/E=64,N=1408,device_name=NVIDIA_B200.json`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #30210 - [Bugfix]: Fix glm46 awq marlin moe wna16 compatibility

- Link: https://github.com/vllm-project/vllm/pull/30210
- Status/date: `merged`, created 2025-12-07, merged 2025-12-09; author `baonudesifeizhai`.
- Diff scope read: `2` files, `+50/-4`; areas: MoE/router, quantization, scheduler/runtime; keywords: config, moe, awq, cuda, marlin, quant.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/fused_moe.py` modified +45/-0 (45 lines); hunks: def get_moe_configs(; def get_moe_wna16_block_config(; symbols: get_moe_configs, _ensure_block_size_k_divisible, get_moe_wna16_block_config, get_moe_wna16_block_config
  - `vllm/model_executor/layers/quantization/moe_wna16.py` modified +5/-4 (9 lines); hunks: def __init__(; def from_config(cls, config: dict[str, Any]) -> "MoeWNA16Config":; symbols: __init__, from_config, get_quant_method, moe_wna16_weight_loader
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/quantization/moe_wna16.py`; keywords observed in patches: config, moe, awq, cuda, marlin, quant. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/quantization/moe_wna16.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #30876 - GLM-4.7 Tool Parser and Doc Update

- Link: https://github.com/vllm-project/vllm/pull/30876
- Status/date: `merged`, created 2025-12-17, merged 2025-12-20; author `zRzRzRzRzRzRzR`.
- Diff scope read: `5` files, `+38/-3`; areas: model wrapper, MoE/router, scheduler/runtime, docs/config; keywords: moe, doc, spec.
- Code diff details:
  - `vllm/tool_parsers/glm47_moe_tool_parser.py` added +23/-0 (23 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: Glm47MoeModelToolParser, __init__
  - `docs/features/tool_calling.md` modified +8/-1 (9 lines); hunks: Supported models:
  - `vllm/tool_parsers/__init__.py` modified +4/-0 (4 lines); hunks: "glm4_moe_tool_parser",
  - `vllm/model_executor/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  - `docs/models/supported_models.md` modified +1/-1 (2 lines); hunks: th {
- Optimization/support interpretation: The concrete diff surface is `vllm/tool_parsers/glm47_moe_tool_parser.py`, `docs/features/tool_calling.md`, `vllm/tool_parsers/__init__.py`; keywords observed in patches: moe, doc, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/tool_parsers/glm47_moe_tool_parser.py`, `docs/features/tool_calling.md`, `vllm/tool_parsers/__init__.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #31386 - [GLM-4.7] GLM Model support for GLM-Lite

- Link: https://github.com/vllm-project/vllm/pull/31386
- Status/date: `merged`, created 2025-12-26, merged 2026-01-19; author `zRzRzRzRzRzRzR`.
- Diff scope read: `9` files, `+1135/-1`; areas: model wrapper, MoE/router, kernel, scheduler/runtime, tests/benchmarks, docs/config; keywords: moe, config, spec, expert, flash, kv, topk, benchmark, processor, quant.
- Code diff details:
  - `vllm/model_executor/models/glm4_moe_lite.py` added +642/-0 (642 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: Glm4MoeLiteMLP, Glm4MoeLite, Glm4LiteMixtureOfExperts, Glm4MoeLiteAttention
  - `vllm/model_executor/models/glm4_moe_lite_mtp.py` added +464/-0 (464 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: SharedHead, __init__, forward, Glm4MoeLiteMultiTokenPredictorLayer
  - `vllm/config/speculative.py` modified +12/-0 (12 lines); hunks: "deepseek_mtp",; def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:; symbols: hf_config_override
  - `tests/models/registry.py` modified +10/-0 (10 lines); hunks: def check_available_online(; def check_available_online(; symbols: check_available_online, check_available_online
  - `vllm/model_executor/models/glm4_moe_mtp.py` modified +2/-1 (3 lines); hunks: # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/glm4_moe_lite.py`, `vllm/model_executor/models/glm4_moe_lite_mtp.py`, `vllm/config/speculative.py`; keywords observed in patches: moe, config, spec, expert, flash, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/glm4_moe_lite.py`, `vllm/model_executor/models/glm4_moe_lite_mtp.py`, `vllm/config/speculative.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #37386 - fix(glm47): improve tool call parsing and content normalization

- Link: https://github.com/vllm-project/vllm/pull/37386
- Status/date: `merged`, created 2026-03-18, merged 2026-03-18; author `karanb192`.
- Diff scope read: `4` files, `+193/-6`; areas: MoE/router, tests/benchmarks; keywords: moe, test, spec.
- Code diff details:
  - `tests/tool_parsers/test_glm47_moe_tool_parser.py` added +168/-0 (168 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: glm47_tokenizer, glm47_tool_parser, mock_request, TestGlm47ExtractToolCalls:
  - `vllm/tool_parsers/glm47_moe_tool_parser.py` modified +16/-2 (18 lines); hunks: # SPDX-License-Identifier: Apache-2.0; class Glm47MoeModelToolParser(Glm4MoeModelToolParser):; symbols: Glm47MoeModelToolParser, __init__
  - `vllm/tool_parsers/glm4_moe_tool_parser.py` modified +6/-1 (7 lines); hunks: def extract_tool_calls(; symbols: extract_tool_calls
  - `tests/tool_parsers/test_glm4_moe_tool_parser.py` modified +3/-3 (6 lines); hunks: def test_extract_tool_calls_no_tools(glm4_moe_tool_parser, mock_request):; def test_extract_tool_calls_no_tools(glm4_moe_tool_parser, mock_request):; symbols: test_extract_tool_calls_no_tools, test_extract_tool_calls_no_tools, test_extract_tool_calls_no_tools
- Optimization/support interpretation: The concrete diff surface is `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`; keywords observed in patches: moe, test, spec. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `tests/tool_parsers/test_glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm47_moe_tool_parser.py`, `vllm/tool_parsers/glm4_moe_tool_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
