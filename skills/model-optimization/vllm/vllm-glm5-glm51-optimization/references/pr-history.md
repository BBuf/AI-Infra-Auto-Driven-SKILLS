# vLLM GLM-5 / 5.1 PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: partially supported on current mainline
- Scope: GLM-5 family adaptation into the DeepSeek runtime plus MTP accuracy
  stabilization

## Landed PRs

### PR #34124 - GLM adaptation

- Link: https://github.com/vllm-project/vllm/pull/34124
- State: merged
- Diff coverage: full diff reviewed, `7` files, `13` additions, `3` deletions
- Motivation:
  - GLM-5 checkpoints expose a `glm_moe_dsa` architecture that was close to the
    DeepSeek MLA/MoE runtime, but vLLM had no registry alias or config
    conversion for it.
  - GLM-5 MTP also needed to be routed through the existing `deepseek_mtp`
    speculative machinery.
- Key implementation:
  - Registers `GlmMoeDsaForCausalLM` in `registry.py` but maps it to the
    `deepseek_v2` module.
  - Adds a trivial `GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)` subclass.
  - Treats `glm_moe_dsa` like `deepseek_mtp` in speculative config override.
  - Extends the arch config converter and initialization/test registry.
  - Changes rope style selection to respect `indexer_rope_interleave`.
- Key code excerpts:

```diff
+class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
+    pass
```

```diff
+    "GlmMoeDsaForCausalLM": ("deepseek_v2", "GlmMoeDsaForCausalLM"),
```

```diff
-        if hf_config.model_type in ("deepseek_v3", "deepseek_v32"):
+        if hf_config.model_type in ("deepseek_v3", "deepseek_v32", "glm_moe_dsa"):
             hf_config.model_type = "deepseek_mtp"
```

```diff
-                is_neox_style=True,
+                is_neox_style=not getattr(config, "indexer_rope_interleave", True),
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_v2.py`,
    `vllm/model_executor/models/registry.py`
  - config/spec decode:
    `vllm/config/speculative.py`,
    `vllm/transformers_utils/model_arch_config_convertor.py`
  - tests: `tests/models/registry.py`, `tests/models/test_initialization.py`
- Validation implications:
  - GLM-5 support should be validated against the DeepSeek runtime, not the
    older `glm4*` files.
  - Rope and speculative decode regressions can affect both GLM-5 and DeepSeek
    because they now share the same path.

### PR #34385 - Fix MTP accuracy for GLM-5

- Link: https://github.com/vllm-project/vllm/pull/34385
- State: merged
- Diff coverage: full diff reviewed, `1` file, `18` additions
- Motivation:
  - GLM-5 MTP startup could succeed while still producing bad logits because the
    target `lm_head` was not propagated into each MTP layer's `shared_head.head`.
- Key implementation:
  - Walks through the draft model layers inside `SpecDecodeBaseProposer`.
  - Replaces every `shared_head.head` with the target language model's
    `lm_head`, not just `self.model.lm_head`.
- Key code excerpts:

```diff
+            inner = getattr(self.model, "model", None)
+            layers = getattr(inner, "layers", None) if inner else None
+            if layers is not None:
+                items = layers.values() if isinstance(layers, nn.ModuleDict) else layers
+                for layer in items:
+                    sh = getattr(layer, "shared_head", None)
+                    if sh is not None and hasattr(sh, "head"):
+                        del sh.head
+                        sh.head = target_language_model.lm_head
```

- Reviewed files:
  - runtime/spec decode: `vllm/v1/spec_decode/eagle.py`
- Validation implications:
  - When GLM-5 MTP produces NaN or zero logits, inspect shared-head weight
    sharing before blaming sampling or tokenizer code.
  - Any future refactor of `eagle.py` can silently regress GLM-5 MTP because the
    fix lives in generic speculative infrastructure, not in a GLM-only module.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# VLLM GLM-5 / GLM-5.1 PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2026-02-09 | [#34124](https://github.com/vllm-project/vllm/pull/34124) | merged | [Model] GLM adaptation | model wrapper, MoE/router, kernel, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/deepseek_v2.py`, `tests/models/registry.py`, `tests/models/test_initialization.py` |
| 2026-02-11 | [#34385](https://github.com/vllm-project/vllm/pull/34385) | merged | [Bugfix] Fix MTP accuracy for GLM-5 | scheduler/runtime | `vllm/v1/spec_decode/eagle.py` |

## Diff Cards

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


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
