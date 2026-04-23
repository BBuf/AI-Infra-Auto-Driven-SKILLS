# GLM-5/5.1 PR History

Evidence sweep:

- SGLang `origin/main`: `b3e6cf60a` (`2026-04-22`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- Manual diff review date: `2026-04-23`
- Searched paths: GLM MoE/NextN files, NSA indexer/backend files, GLM-5 docs/snippets, registered GLM-5 tests.
- Searched PR terms: `GLM-5`, `GLM5`, `GLM-5.1`, `GLM51`, `glm5`, `glm51`, `GlmMoeDsa`.

## Runtime Surfaces

- `python/sglang/srt/models/glm4_moe.py`
- `python/sglang/srt/models/glm4_moe_nextn.py`
- `python/sglang/srt/models/deepseek_nextn.py`
- `python/sglang/srt/models/deepseek_v2.py`
- `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`
- `python/sglang/srt/layers/attention/nsa/`
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/entrypoints/openai/serving_chat.py`
- `docs_new/cookbook/autoregressive/GLM/GLM-5.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-5.1.mdx`
- `test/registered/8-gpu-models/test_dsa_models_basic.py`
- `test/registered/8-gpu-models/test_dsa_models_mtp.py`
- `test/registered/8-gpu-models/test_glm_51_fp8.py`
- `test/registered/gb300/test_glm5_fp8.py`
- `test/registered/gb300/test_glm5_nvfp4.py`
- `test/registered/amd/accuracy/`
- `test/registered/amd/perf/`

## Diff-Reviewed PR Cards

### PR #18521 - Support GlmMoeDsaForCausalLM

- Link: https://github.com/sgl-project/sglang/pull/18521
- State: merged at `2026-02-10T07:20:10Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `462` lines, `3` files.
- Motivation: GLM-5 uses a DSA/NSA architecture close enough to DeepSeek V3.2 that the first support path should reuse the existing `DeepseekV2ForCausalLM` and NSA backend instead of copying a GLM-specific stack. The PR also had to handle RoPE parameter differences and speculative draft-model architecture rewriting so GLM-5 could enter the same DSA and NextN machinery as DeepSeek.
- Key implementation: `is_deepseek_nsa()` recognizes `GlmMoeDsaForCausalLM`; `ModelConfig._config_draft_model()` maps GLM DSA draft models to `DeepseekV3ForCausalLMNextN`; `GlmMoeDsaForCausalLM` is added as a subclass of `DeepseekV2ForCausalLM`; server argument handling adds GLM DSA to NSA backend auto-selection, deterministic inference, speculative decoding, and auto speculative parameter choices. Earlier commits in the same PR make `Indexer` accept dynamic `is_neox_style` and support transformers v4/v5 RoPE parameter layouts.
- Key code excerpts:

```diff
+            "GlmMoeDsaForCausalLM",
         ]
         and getattr(config, "index_topk", None) is not None
```

```python
class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    pass

EntryClass = [Glm4MoeForCausalLM, GlmMoeDsaForCausalLM]
```

```diff
+                if model_arch == "GlmMoeDsaForCausalLM" and is_blackwell_supported():
+                    envs.SGLANG_NSA_FORCE_MLA.set(True)
```

- Reviewed files: `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`.
- Validation implications: GLM-5 launch should default to NSA attention, support DSA shape derivation, and cover speculative/MTP paths through the DeepSeek NextN adapter. Blackwell runs must pay attention to the forced sparse-MLA behavior that was later refined by #20062.

### PR #18804 - Fix GLM-5 fused shared expert

- Link: https://github.com/sgl-project/sglang/pull/18804
- State: merged at `2026-02-16T19:50:39Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `131` lines, `1` file.
- Motivation: after #18521, `GlmMoeDsaForCausalLM` inherited DeepSeek behavior but did not override the fused shared-expert count hook. GLM-5 therefore risked using the wrong shared-expert fusion metadata when loading/running the MoE path.
- Key implementation: `GlmMoeDsaForCausalLM.determine_num_fused_shared_experts()` delegates to the DeepSeek base implementation with the GLM class name. The intermediate review commits tried `self.__class__.__name__` but the final patch pins the explicit architecture string.
- Key code excerpt:

```python
class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    def determine_num_fused_shared_experts(self):
        super().determine_num_fused_shared_experts("GlmMoeDsaForCausalLM")
```

- Reviewed files: `python/sglang/srt/models/glm4_moe.py`.
- Validation implications: GLM-5 MoE tests must verify shared-expert routing/fusion, not only that the server boots. This card is a loader/runtime correctness fix, not a docs-only change.

### PR #18911 - AMD GLM-5 day-0 nightly test

- Link: https://github.com/sgl-project/sglang/pull/18911
- State: merged at `2026-02-25T03:39:17Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `1274` lines, `5` files.
- Motivation: GLM-5 needed early ROCm coverage. The diff shows two concerns: HIP RoPE fallback must avoid CUDA-only JIT/tvm paths, and AMD nightly should actually run GLM-5 accuracy on MI30x/MI35x instead of relying on NVIDIA-only DSA tests.
- Key implementation: `RotaryEmbedding.forward_hip()` is added and finally implemented as `return self.forward_native(*args, **kwargs)` so subclasses with different `forward_native()` signatures still work. The PR adds AMD/ROCm workflow entries and `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` plus `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` for 8-GPU GLM-5 evaluation.
- Key code excerpts:

```python
def forward_hip(self, *args, **kwargs):
    """HIP/ROCm implementation."""
    return self.forward_native(*args, **kwargs)
```

```python
GLM5_MODEL_PATH = "zai-org/GLM-5-FP8"
```

- Reviewed files: `.github/workflows/nightly-test-amd.yml`, `.github/workflows/nightly-test-amd-rocm720.yml`, `python/sglang/srt/layers/rotary_embedding.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`.
- Validation implications: GLM-5 AMD regressions should include RoPE on HIP and the day-0 GSM8K accuracy path. Subclass compatibility matters because GLM/VLM RoPE variants can have different native signatures.

### PR #20062 - Control dense-attention threshold for V3.2/GLM-5

- Link: https://github.com/sgl-project/sglang/pull/20062
- State: merged at `2026-03-09T21:36:10Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `588` lines, `6` files.
- Motivation: #18521 used a binary `SGLANG_NSA_FORCE_MLA` switch to disable MHA one-shot for GLM DSA on Blackwell. That was too coarse. DSA models need a tunable threshold: short prefill can use dense MHA for speed, but longer KV lengths should switch to sparse MLA to avoid accuracy/performance pathologies. GLM-5 on Blackwell forces the threshold to zero.
- Key implementation: `SGLANG_NSA_DENSE_ATTN_KV_LEN_THRESHOLD` is introduced as an integer environment variable. `server_args.py` sets it to zero for `GlmMoeDsaForCausalLM` on Blackwell, or to the model `index_topk` when not manually set. `nsa_backend.py` replaces the old backend-specific `mha_max_kv_len` with this env threshold in `set_nsa_prefill_impl()`.
- Key code excerpts:

```python
SGLANG_NSA_DENSE_ATTN_KV_LEN_THRESHOLD = EnvInt(2048)
```

```python
if model_arch == "GlmMoeDsaForCausalLM" and is_blackwell_supported():
    envs.SGLANG_NSA_DENSE_ATTN_KV_LEN_THRESHOLD.set(0)
```

```diff
-                and max_kv_len <= mha_max_kv_len
+                and max_kv_len
+                <= envs.SGLANG_NSA_DENSE_ATTN_KV_LEN_THRESHOLD.get()
```

- Reviewed files: `python/sglang/srt/environ.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `docs/references/environment_variables.md`.
- Validation implications: GLM-5/5.1 Blackwell tests must check the sparse-MLA path. Hopper/AMD runs should verify that the threshold defaults to `index_topk` unless manually overridden.

### PR #21710 - AMD GLM-5-FP8 performance benchmarks

- Link: https://github.com/sgl-project/sglang/pull/21710
- State: merged at `2026-04-08T05:43:14Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `537` lines, `6` files.
- Motivation: GLM-5-FP8 already had AMD accuracy coverage, but there was no nightly throughput/latency benchmark for MI30x and MI35x. The PR body explicitly makes performance non-blocking while keeping accuracy blocking, so regressions can be observed without hiding correctness failures.
- Key implementation: the AMD workflows add performance steps after accuracy. The accuracy configs switch to `zai-org/GLM-5-FP8` and add `--reasoning-parser glm45 --tool-call-parser glm47`. New perf tests use `bench_one_batch`, `--kv-cache-dtype fp8_e4m3`, and AMD tuning env such as `SGLANG_USE_AITER=1`.
- Key code excerpt:

```yaml
- name: Performance Test ROCm 7.2 (8-GPU GLM-5)
  timeout-minutes: 120
  continue-on-error: true
  run: |
    python3 run_suite.py --hw amd --suite nightly-perf-8-gpu-glm5 --nightly
```

```python
model_path="zai-org/GLM-5-FP8",
other_args=[
    "--reasoning-parser", "glm45",
    "--tool-call-parser", "glm47",
]
```

- Reviewed files: AMD nightly workflow files, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`.
- Validation implications: GLM-5 command guidance should keep `glm45`/`glm47` parsers aligned with AMD tests. Performance dashboards should distinguish accuracy gating from non-blocking perf alerts.

### PR #21773 - AMD GLM-5-MXFP4 MI35x accuracy/perf tests

- Link: https://github.com/sgl-project/sglang/pull/21773
- State: merged at `2026-04-15T01:55:36Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `863` lines, `4` files.
- Motivation: GLM-5 MXFP4/Quark checkpoints needed a separate MI35x validation lane from GLM-5-FP8. The workflows are reshaped so GLM-5-MXFP4 has its own job filter entry and no longer conflates FP8 GLM-5 and GLM-5.1 jobs.
- Key implementation: the PR adds `test_glm5_mxfp4_eval_mi35x.py` and `test_glm5_mxfp4_perf_mi35x.py`, wires workflow entries named `nightly-8-gpu-mi35x-glm5-mxfp4`, and runs both accuracy and perf under `SGLANG_USE_AITER=1`. The perf path directly invokes the MI35x perf script with a longer timeout.
- Key code excerpt:

```yaml
nightly-8-gpu-mi35x-glm5-mxfp4-rocm720:
  runs-on: linux-mi35x-gpu-8
```

```yaml
python3 registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py
```

- Reviewed files: `.github/workflows/nightly-test-amd.yml`, `.github/workflows/nightly-test-amd-rocm720.yml`, `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py`.
- Validation implications: GLM-5 MXFP4 should be tracked independently from GLM-5 FP8 and GLM-5.1 FP8. The Quark/MXFP4 loader fixes in #22543 and MTP fixes in #23219 should be validated against this lane.

### PR #22179 - Improve DeepSeek V3.2/GLM-5 documentation

- Link: https://github.com/sgl-project/sglang/pull/22179
- State: merged at `2026-04-06T06:26:43Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `127` lines, `1` file.
- Motivation: GLM-5 shares the DSA/NSA usage surface with DeepSeek V3.2 but has different parser choices. The existing old docs needed to make that relationship explicit and document the adaptive short-sequence MHA behavior and IndexCache recommendation for GLM-5.
- Key implementation: `docs/basic_usage/deepseek_v32.md` now states that GLM-5 can use the DSA instructions by replacing the model with `zai-org/GLM-5-FP8`, except for reasoning/tool parsers. It documents short-sequence MHA prefill, backend choices, and an IndexCache `index_topk_pattern` override for GLM-5. Note that the doc hunk names `SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`, while #20062 introduced `SGLANG_NSA_DENSE_ATTN_KV_LEN_THRESHOLD`; future docs should reconcile that naming before copying it.
- Key code excerpt:

```diff
-To server GLM-5, just replace the `--model` argument with `zai-org/GLM-5-FP8`.
+To serve GLM-5, just replace the `--model` argument with `zai-org/GLM-5-FP8`.
```

```markdown
For **GLM-5** model, we recommend appending
`--json-model-override-args '{"index_topk_pattern": "..."}'`
```

- Reviewed files: `docs/basic_usage/deepseek_v32.md`.
- Validation implications: new GLM-5 docs must preserve `--tool-call-parser glm47`, `--reasoning-parser glm45`, NSA backend flags, and IndexCache caveats. Verify the dense-attention env-var name against current code.

### PR #22285 - Add CI tests for GLM-5

- Link: https://github.com/sgl-project/sglang/pull/22285
- State: merged at `2026-04-08T08:05:36Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `8911` lines, `2` files; the renamed DeepSeek/GLM shared test files and GLM-added classes were reviewed manually.
- Motivation: GLM-5 should not only have docs and AMD-specific tests; it needs the same H200 8-GPU DSA regression coverage as DeepSeek V3.2, including TP, DP attention, and MTP/spec-v2 variants.
- Key implementation: DeepSeek V3.2 test files are renamed to DSA model test files. GLM-5 DP/TP classes launch `zai-org/GLM-5-FP8` with `--tp 8`, optional `--dp 8 --enable-dp-attention`, and multithreaded weight loading. MTP classes add EAGLE settings, check GSM8K score, read `avg_spec_accept_length` from `/server_info`, and assert acceptance length and speed.
- Key code excerpts:

```python
GLM5_MODEL_PATH = "zai-org/GLM-5-FP8"
```

```python
other_args = [
    "--trust-remote-code",
    "--tp", "8",
    "--dp", "8",
    "--enable-dp-attention",
]
```

```python
self.assertGreater(metrics["score"], 0.94)
self.assertGreater(avg_spec_accept_length, 2.7)
```

- Reviewed files: `test/registered/8-gpu-models/test_dsa_models_basic.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`.
- Validation implications: GLM-5 core regressions should include both non-MTP and MTP/spec-v2 lanes, and they should inspect speculative acceptance, not only final accuracy.

### PR #22314 - AMD GLM-5 FP8 KV quant dispatch on MI300

- Link: https://github.com/sgl-project/sglang/pull/22314
- State: merged at `2026-04-08T04:16:02Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `121` lines, `1` file.
- Motivation: the memory-pool MLA KV write path mixed NVIDIA FP8 KV-cache store logic with HIP raw MLA KV layout. On MI300/ROCm, GLM-5 FP8 KV should use the HIP fused BF16/FP16-to-FP8 paged KV write instead of the NVIDIA path that quantizes `k_nope` and `k_rope` into the byte/scales layout.
- Key implementation: `set_mla_kv_buffer()` checks `_is_hip and self.use_nsa and self.dtype == fp8_dtype` before `self.nsa_kv_cache_store_fp8`. That HIP branch calls `set_mla_kv_buffer_triton_fp8_quant()` directly with `cache_k_nope`, `cache_k_rope`, and imported `fp8_dtype`; non-HIP keeps the separate quantize/write path.
- Key code excerpt:

```python
if _is_hip and self.use_nsa and self.dtype == fp8_dtype:
    set_mla_kv_buffer_triton_fp8_quant(
        self.kv_buffer[layer_id - self.start_layer],
        loc,
        cache_k_nope,
        cache_k_rope,
        fp8_dtype,
    )
elif self.nsa_kv_cache_store_fp8:
    cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(...)
```

- Reviewed files: `python/sglang/srt/mem_cache/memory_pool.py`.
- Validation implications: GLM-5 FP8 KV tests on MI300/MI35x should exercise NSA KV-cache writes with `fp8_e4m3` and compare against a BF16/no-FP8-KV baseline.

### PR #22336 - AMD GLM-5.1-FP8 nightly tests

- Link: https://github.com/sgl-project/sglang/pull/22336
- State: merged at `2026-04-09T05:57:43Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `1485` lines, `6` files.
- Motivation: GLM-5.1-FP8 is a larger MoE DSA model and needs separate AMD MI30x/MI35x coverage from GLM-5-FP8. Its launch uses TP=8 and EP=8, matching the expert-parallel shape used by other large AMD MoE jobs.
- Key implementation: the AMD workflows gain `nightly-8-gpu-glm51` and `nightly-8-gpu-mi35x-glm51` jobs. New accuracy and perf tests launch `zai-org/GLM-5.1-FP8` with `--tp 8 --ep-size 8`, `--nsa-prefill-backend tilelang`, `--nsa-decode-backend tilelang`, `--reasoning-parser=glm45`, and `--tool-call-parser=glm47`; perf adds `--kv-cache-dtype fp8_e4m3` and MI35x env tuning.
- Key code excerpt:

```python
model_path="zai-org/GLM-5.1-FP8"
other_args=[
    "--tp", "8",
    "--ep-size", "8",
    "--reasoning-parser=glm45",
    "--tool-call-parser=glm47",
]
```

- Reviewed files: AMD nightly workflows, `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py`.
- Validation implications: GLM-5.1 documentation should mention EP=8 where relevant. Accuracy and performance failures should be diagnosed separately for MI30x and MI35x because the perf env differs.

### PR #22399 - GLM-5.1 nightly tests and Qwen3.5 model update

- Link: https://github.com/sgl-project/sglang/pull/22399
- State: merged at `2026-04-09T00:04:57Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `225` lines, `3` files.
- Motivation: NVIDIA H200/B200 and GB300 CI needed GLM-5.1-FP8 coverage, while GLM-5 NVFP4 still pointed at GLM-5 rather than a nonexistent GLM-5.1 NVFP4 checkpoint.
- Key implementation: `test_glm_51_fp8.py` adds H200/B200 `nightly-8-gpu-common` variants for TP8, TP8+DP8, and TP8+DP8+MTP with `SGLANG_ENABLE_SPEC_V2=1`. GB300 GLM-5 FP8 tests update their model path to `zai-org/GLM-5.1-FP8`; a second commit reverts the NVFP4 test name/docstring back to GLM-5 because GLM-5.1 NVFP4 does not exist.
- Key code excerpts:

```python
GLM_51_FP8_MODEL_PATH = "zai-org/GLM-5.1-FP8"
COMMON_ARGS = [
    "--reasoning-parser=glm45",
    "--tool-call-parser=glm47",
]
```

```python
variant="TP8+DP8+MTP",
env={"SGLANG_ENABLE_SPEC_V2": "1"},
```

- Reviewed files: `test/registered/8-gpu-models/test_glm_51_fp8.py`, `test/registered/gb300/test_glm5_fp8.py`, `test/registered/gb300/test_glm5_nvfp4.py`.
- Validation implications: GLM-5.1 FP8 is the H200/B200/GB300 path; GLM-5 NVFP4 remains GLM-5. Do not rename NVFP4 docs/tests to GLM-5.1 unless a real checkpoint exists.

### PR #22543 - GLM-5/5.1 MXFP4 checkpoint inference compatibility

- Link: https://github.com/sgl-project/sglang/pull/22543
- State: merged at `2026-04-14T06:56:49Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `122` lines, `3` files.
- Motivation: MXFP4/Quark GLM checkpoints share DeepSeek weight-loader infrastructure but should not run DeepSeek-V3-specific Quark post-load transforms. They also need `gate_up_proj` packing for Quark fused MLP weights.
- Key implementation: the DeepSeek weight loader only applies `quark_post_load_weights(..., "mxfp4")` when `self.config.architectures[0] == "DeepseekV3ForCausalLM"`, explicitly avoiding `GlmMoeDsaForCausalLM`. `_get_quantization_config()` adds `{"gate_up_proj": ["gate_proj", "up_proj"]}` to `packed_modules_mapping` when `model_config.quantization == "quark"`. The server arg default handler strips device indices such as `cuda:0` down to `cuda`.
- Key code excerpts:

```python
if model_config.quantization == "quark":
    packed_modules_mapping.update({"gate_up_proj": ["gate_proj", "up_proj"]})
```

```python
and self.config.architectures
and self.config.architectures[0] == "DeepseekV3ForCausalLM"
```

- Reviewed files: `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/server_args.py`.
- Validation implications: GLM-5/5.1 MXFP4 startup should verify gate/up fused weight loading and ensure no DeepSeek-only Quark post-load path mutates GLM DSA weights.

### PR #22595 - Normalize tool message content for GLM5.1 chat template

- Link: https://github.com/sgl-project/sglang/pull/22595
- State: merged at `2026-04-16T08:48:38Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `191` lines, `2` files.
- Motivation: OpenAI clients can send tool role content as content-part arrays such as `[{"type": "text", "text": "..."}]`, while GLM-5/GLM-5.1 chat templates expect tool messages to be strings. The result was invisible tool output and repeated tool calls instead of a final natural-language answer.
- Key implementation: `normalize_tool_content(role, content)` is added to `serving_chat.py`. It only flattens `role == "tool"` lists whose items are all strings or OpenAI text parts, joins them with spaces, and preserves lists with non-text semantic fields. Unit tests cover flattening, mixed string/dict text parts, empty lists, non-tool roles, and preserving structured tool lists.
- Key code excerpt:

```python
def normalize_tool_content(role: str, content):
    if role != "tool" or not isinstance(content, list):
        return content
    is_openai_text_parts = all(
        (isinstance(p, dict) and p.get("type") == "text") or isinstance(p, str)
        for p in content
    )
    if is_openai_text_parts:
        return " ".join(p.get("text", "") if isinstance(p, dict) else p for p in content)
    return content
```

- Reviewed files: `python/sglang/srt/entrypoints/openai/serving_chat.py`, `test/registered/openai_server/basic/test_serving_chat.py`.
- Validation implications: GLM-5.1 tool-calling tests should include tool result content as OpenAI text-part arrays and ensure the model produces a final answer instead of repeating calls.

### PR #22712 - NPU GLM-5 running guide

- Link: https://github.com/sgl-project/sglang/pull/22712
- State: merged at `2026-04-13T14:53:24Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `33` lines, `1` file.
- Motivation: Ascend GLM-5 deployment docs previously told users to update transformers from main. The GLM-5 best-practice path needed a pinned version to avoid accidental breakage from transformer mainline changes.
- Key implementation: `docs/platforms/ascend/ascend_npu_glm5_examples.md` now recommends transformers `5.3.0`, either from PyPI or the GitHub `v5.3.0` tag.
- Key code excerpt:

```diff
-pip install git+https://github.com/huggingface/transformers.git
+pip install transformers==5.3.0
+pip install git+https://github.com/huggingface/transformers.git@v5.3.0
```

- Reviewed files: `docs/platforms/ascend/ascend_npu_glm5_examples.md`.
- Validation implications: NPU docs and smoke tests should pin transformers consistently with this guide instead of relying on `main`.

### PR #22850 - AMD NSA indexer kernel reduction

- Link: https://github.com/sgl-project/sglang/pull/22850
- State: merged at `2026-04-19T07:18:12Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `141` lines, `1` file.
- Motivation: AMD DSA/GLM-5 NSA indexer still had extra kernels and dtype conversions around `weights_proj` and index-K cache storage. This hurt the GLM-5/DeepSeek V3.2 AMD path where NSA indexer overhead is visible.
- Key implementation: `weights_proj` parameters are now BF16 on all platforms and HIP returns BF16 directly because multiplying by `q_scale` promotes back to FP32. When `SGLANG_USE_AITER` is active, `_store_index_k_cache()` calls `aiter.ops.cache.indexer_k_quant_and_cache` to fuse quantization and cache write, reshaping the uint8 buffer to the FP8 view required by the kernel.
- Key code excerpts:

```python
weights, _ = self.weights_proj(x)
if _is_hip:
    # Return bf16; multiplying with q_scale promotes back to fp32.
    return weights
```

```python
if _use_aiter:
    buf = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(layer_id=layer_id)
    kv_cache = buf.unsqueeze(1).view(fp8_dtype)
    indexer_k_quant_and_cache(key, kv_cache, out_loc, self.block_size, self.scale_fmt)
    return
```

- Reviewed files: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`.
- Validation implications: AMD GLM-5/GLM-5.1 perf should be checked with `SGLANG_USE_AITER=1`, FP8 index-K cache storage, and a non-AITER fallback to catch accuracy drift.

### PR #23219 - Enable MTP for GLM-5-MXFP4

- Link: https://github.com/sgl-project/sglang/pull/23219
- State: merged at `2026-04-20T23:09:08Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `121` lines, `1` file.
- Motivation: GLM-5-MXFP4 uses Quark quantization and shared DeepSeek NextN code. The draft `eh_proj` projection and MTP layer quantization needed to respect Quark checkpoint layout and `exclude_layers`; otherwise MTP could try to quantize or load the draft layer incorrectly.
- Key implementation: `deepseek_nextn.py` uses `ReplicatedLinear` for `eh_proj` when `quant_config.get_name() == "quark"`, and the forward path handles its `(output, bias)` return. Before constructing `DeepseekModelNextN`, the PR checks whether the MTP layer prefix is listed in Quark `exclude_layers`; if so it sets `nextn_quant_config = None`.
- Key code excerpts:

```python
if quant_config is not None and quant_config.get_name() == "quark":
    self.eh_proj = ReplicatedLinear(
        2 * config.hidden_size,
        config.hidden_size,
        bias=False,
        quant_config=quant_config,
        prefix=add_prefix("eh_proj", prefix),
    )
```

```python
if should_ignore_layer(mapped_prefix, nextn_quant_config.exclude_layers):
    nextn_quant_config = None
```

- Reviewed files: `python/sglang/srt/models/deepseek_nextn.py`.
- Validation implications: GLM-5-MXFP4 MTP must be tested separately from FP8 MTP. The regression should check Quark `exclude_layers`, `eh_proj` loading, and output quality with EAGLE settings.

## Cookbook Evidence

- sgl-cookbook PRs are documentation-parity inputs only until their diffs are reviewed with the same card standard. Do not cite them as runtime evidence without opening the cookbook diff.

## Validation Notes

- GLM-5/5.1 is a shared DSA/NSA lane. Any change to `deepseek_v2.py`, `deepseek_nextn.py`, `nsa_backend.py`, or `nsa_indexer.py` can affect DeepSeek V3.2 and GLM simultaneously.
- Preserve parser defaults in examples: `--tool-call-parser glm47` and `--reasoning-parser glm45`.
- Keep GLM-5 FP8, GLM-5 MXFP4, GLM-5 NVFP4, and GLM-5.1 FP8 validation separate; #22399 explicitly avoids pretending a GLM-5.1 NVFP4 checkpoint exists.
- For AMD, distinguish correctness CI from non-blocking performance CI and include `SGLANG_USE_AITER=1` lanes where the diff depends on AITER.
