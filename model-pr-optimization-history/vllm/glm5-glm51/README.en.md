# vLLM GLM-5 / 5.1 Support and PR History

This note tracks the landed GLM-5 / 5.1 path in vLLM at commit
`0f7be0f2f76814f80f9091220a5fbbb53912ad00`.

- Status: partially supported on current mainline

## Key Conclusions

- GLM-5 support is currently an adaptation layer on top of the DeepSeek-V2/V3
  runtime, not an independent `glm5.py` implementation.
- The two key landed steps are:
  `#34124` for architecture/config adaptation and `#34385` for MTP accuracy.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/deepseek_v2.py`
- `vllm/vllm/model_executor/models/registry.py`
- `vllm/vllm/config/speculative.py`
- `vllm/vllm/transformers_utils/model_arch_config_convertor.py`
- `vllm/vllm/v1/spec_decode/eagle.py`

## Landed PRs

- [#34124](https://github.com/vllm-project/vllm/pull/34124)
  `GLM adaptation`
  Diff reviewed: `7` files, `13` additions, `3` deletions.
  Adds `GlmMoeDsaForCausalLM` as a DeepSeek-V2-derived alias, extends
  speculative config conversion, and respects `indexer_rope_interleave`.
- [#34385](https://github.com/vllm-project/vllm/pull/34385)
  `Fix MTP accuracy for GLM-5`
  Diff reviewed: `1` file, `18` additions.
  Shares the target `lm_head` into MTP `shared_head.head` so GLM-5 draft logits
  stop producing NaNs or uninitialized outputs.

## Current Contract

If GLM-5 breaks, inspect the DeepSeek-based MLA/MoE runtime and speculative
decode stack first. Do not assume the older `glm4*` files are the source of
truth for GLM-5 behavior.

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `GLM-5 / GLM-5.1` against `vllm-project/vllm` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2026-02-09 | [#34124](https://github.com/vllm-project/vllm/pull/34124) | merged | [Model] GLM adaptation | model wrapper, MoE/router, kernel, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/deepseek_v2.py`, `tests/models/registry.py`, `tests/models/test_initialization.py` |
| 2026-02-11 | [#34385](https://github.com/vllm-project/vllm/pull/34385) | merged | [Bugfix] Fix MTP accuracy for GLM-5 | scheduler/runtime | `vllm/v1/spec_decode/eagle.py` |

### File-level PR diff reading notes

### PR #34124 - [Model] GLM adaptation

- Link: https://github.com/vllm-project/vllm/pull/34124
- Status/date: `merged`, created 2026-02-09, merged 2026-02-09; author `jeejeelee`.
- Diff scope read: `7` files, `+13/-3`; areas: model wrapper, MoE/router, kernel, scheduler/runtime, tests/benchmarks, docs/config; keywords: moe, config, kv, spec, test, benchmark, cache, flash, mla.
- Code diff details:
  - `vllm/model_executor/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: def __init__(; class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):; symbols: __init__, DeepseekV3ForCausalLM, GlmMoeDsaForCausalLM, get_spec_layer_idx_from_weight_name
  - `tests/models/registry.py` modified +3/-0 (3 lines); hunks: def check_available_online(; symbols: check_available_online
  - `tests/models/test_initialization.py` modified +1/-1 (2 lines); hunks: def _initialize_kv_caches_v1(self, vllm_config):; symbols: _initialize_kv_caches_v1
  - `vllm/config/speculative.py` modified +1/-1 (2 lines); hunks: def compute_hash(self) -> str:; symbols: compute_hash, hf_config_override
  - `benchmarks/kernels/benchmark_moe.py` modified +1/-0 (1 lines); hunks: def get_model_params(config):; symbols: get_model_params
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/deepseek_v2.py`, `tests/models/registry.py`, `tests/models/test_initialization.py`; keywords observed in patches: moe, config, kv, spec, test, benchmark. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/deepseek_v2.py`, `tests/models/registry.py`, `tests/models/test_initialization.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #34385 - [Bugfix] Fix MTP accuracy for GLM-5

- Link: https://github.com/vllm-project/vllm/pull/34385
- Status/date: `merged`, created 2026-02-11, merged 2026-02-12; author `mgoin`.
- Diff scope read: `1` files, `+18/-0`; areas: scheduler/runtime; keywords: eagle, spec.
- Code diff details:
  - `vllm/v1/spec_decode/eagle.py` modified +18/-0 (18 lines); hunks: def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:; symbols: _maybe_share_lm_head, dummy_run
- Optimization/support interpretation: The concrete diff surface is `vllm/v1/spec_decode/eagle.py`; keywords observed in patches: eagle, spec. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/v1/spec_decode/eagle.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 2; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
