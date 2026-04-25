# vLLM DeepSeek V3 / R1 Support and PR History

This note tracks the vLLM runtime, key PRs, and remaining risk areas for DeepSeek V3 / R1.

- Status: supported on current mainline

## Key Conclusions

- `DeepseekV2ForCausalLM` / `DeepseekV3ForCausalLM` remain the shared runtime for V3 and R1.
- The highest-risk regressions cluster around packed module mapping, quantized MLA/MoE weight loading, LoRA, and MTP draft paths.
- R1 validation should split BF16, FP8/ModelOpt, and compressed-tensors or ROCm lanes.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/deepseek_v2.py`
- `vllm/vllm/model_executor/models/deepseek_eagle.py`
- `vllm/vllm/model_executor/models/deepseek_eagle3.py`
- `vllm/vllm/model_executor/models/deepseek_mtp.py`

## Landed PRs

- [#22352](https://github.com/vllm-project/vllm/pull/22352) `Add missing `packed_modules_mapping` to `DeepseekV2ForCausalLM``: Fixed quantized and packed-weight loading for DeepSeek V2/V3/R1 style checkpoints.
- [#23971](https://github.com/vllm-project/vllm/pull/23971) `Add LoRA support for DeepSeek models (V2, V3, R1-0528)`: Enabled adapter injection on the DeepSeek family rather than only base dense models.
- [#29545](https://github.com/vllm-project/vllm/pull/29545) `Fix DeepSeek R1 MTP weight loading`: Hardened R1 NextN / MTP draft loading after launch failures on draft weights.
- [#36247](https://github.com/vllm-project/vllm/pull/36247) `Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x`: Closed a production ROCm gap for compressed-tensors DeepSeek-R1 deployment.

## Open PR Radar

- No pinned open PR here; re-run PR search before claiming new support.

## Matching Skill

- `skills/model-optimization/vllm/vllm-deepseek-v3-r1-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-deepseek-v3-r1-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `DeepSeek V3 / R1` against `vllm-project/vllm` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-08-06 | [#22352](https://github.com/vllm-project/vllm/pull/22352) | merged | [Bugfix] Add missing `packed_modules_mapping` to `DeepseekV2ForCausalLM` | model wrapper, scheduler/runtime | `vllm/model_executor/models/deepseek_v2.py` |
| 2025-08-29 | [#23971](https://github.com/vllm-project/vllm/pull/23971) | merged | Add LoRA support for DeepSeek models (V2, V3, R1-0528) | model wrapper, scheduler/runtime, docs/config | `vllm/model_executor/models/deepseek.py`, `docs/models/supported_models.md`, `vllm/model_executor/models/deepseek_v2.py` |
| 2025-11-26 | [#29545](https://github.com/vllm-project/vllm/pull/29545) | merged | [Bugfix] Fix DeepSeek R1 MTP weight loading | model wrapper, scheduler/runtime | `vllm/model_executor/models/deepseek_mtp.py` |
| 2026-03-06 | [#36247](https://github.com/vllm-project/vllm/pull/36247) | merged | [Bugfix] Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x | model wrapper, scheduler/runtime | `vllm/model_executor/models/deepseek_v2.py` |

### File-level PR diff reading notes

### PR #22352 - [Bugfix] Add missing `packed_modules_mapping` to `DeepseekV2ForCausalLM`

- Link: https://github.com/vllm-project/vllm/pull/22352
- Status/date: `merged`, created 2025-08-06, merged 2025-08-07; author `fxmarty-amd`.
- Diff scope read: `1` files, `+16/-0`; areas: model wrapper, scheduler/runtime; keywords: config, expert, kv, lora, quant.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +16/-0 (16 lines); hunks: def forward(; symbols: forward, DeepseekV2ForCausalLM, __init__
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/deepseek_v2.py`; keywords observed in patches: config, expert, kv, lora, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23971 - Add LoRA support for DeepSeek models (V2, V3, R1-0528)

- Link: https://github.com/vllm-project/vllm/pull/23971
- Status/date: `merged`, created 2025-08-29, merged 2025-08-30; author `sadeghja1070`.
- Diff scope read: `3` files, `+12/-7`; areas: model wrapper, scheduler/runtime, docs/config; keywords: kv, lora, config, doc, expert, moe.
- Code diff details:
  - `vllm/model_executor/models/deepseek.py` modified +6/-2 (8 lines); hunks: from vllm.model_executor.sampling_metadata import SamplingMetadata; def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights, DeepseekForCausalLM, DeepseekForCausalLM, __init__
  - `docs/models/supported_models.md` modified +3/-3 (6 lines); hunks: th {
  - `vllm/model_executor/models/deepseek_v2.py` modified +3/-2 (5 lines); hunks: from vllm.model_executor.sampling_metadata import SamplingMetadata; def forward(; symbols: forward, DeepseekV2ForCausalLM, DeepseekV2ForCausalLM
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/deepseek.py`, `docs/models/supported_models.md`, `vllm/model_executor/models/deepseek_v2.py`; keywords observed in patches: kv, lora, config, doc, expert, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/deepseek.py`, `docs/models/supported_models.md`, `vllm/model_executor/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #29545 - [Bugfix] Fix DeepSeek R1 MTP weight loading

- Link: https://github.com/vllm-project/vllm/pull/29545
- Status/date: `merged`, created 2025-11-26, merged 2025-12-02; author `MatthewBonanni`.
- Diff scope read: `1` files, `+11/-0`; areas: model wrapper, scheduler/runtime; keywords: expert.
- Code diff details:
  - `vllm/model_executor/models/deepseek_mtp.py` modified +11/-0 (11 lines); hunks: def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:; def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str; symbols: load_weights, load_weights
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/deepseek_mtp.py`; keywords observed in patches: expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/deepseek_mtp.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #36247 - [Bugfix] Fix compressed-tensors quantization failure for DeepSeek-R1 on MI300x

- Link: https://github.com/vllm-project/vllm/pull/36247
- Status/date: `merged`, created 2026-03-06, merged 2026-03-07; author `vllmellm`.
- Diff scope read: `1` files, `+2/-2`; areas: model wrapper, scheduler/runtime; keywords: config, kv, lora, quant.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: def _min_latency_fused_qkv_a_proj_fake(; def __init__(; symbols: _min_latency_fused_qkv_a_proj_fake, DeepSeekV2FusedQkvAProj, DeepSeekV2FusedQkvAProjLinear, __init__
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/deepseek_v2.py`; keywords observed in patches: config, kv, lora, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 4; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
