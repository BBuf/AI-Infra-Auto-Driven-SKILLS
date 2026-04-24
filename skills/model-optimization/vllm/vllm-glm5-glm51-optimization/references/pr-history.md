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
  - Extends the arch config convertor and initialization/test registry.
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
