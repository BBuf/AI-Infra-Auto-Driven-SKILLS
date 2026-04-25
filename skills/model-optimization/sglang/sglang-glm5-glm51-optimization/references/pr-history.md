# GLM-5/5.1 PR History

Evidence sweep:

- SGLang `origin/main`: `bca3dd958` (`2026-04-24`)
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

### PR #23060 - Fix dynamic chunking profiling crash on GLM-5 models

- Link: https://github.com/sgl-project/sglang/pull/23060
- State: merged at `2026-04-23T11:30:57Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `30` lines, `1` file.
- Motivation: the pipeline-parallel dynamic chunking profiling path builds a synthetic `ForwardBatch` before calling `model_runner.forward()`. GLM-5 DSA/DP-attention code depends on the thread-local `is_extend_in_batch` flag, so profiling could crash or enter the wrong attention path when that flag was not set.
- Key implementation: `scheduler_pp_mixin.py` imports `set_is_extend_in_batch` and records whether the profiling batch is an extend batch immediately after `ForwardBatch.init_new(...)`.
- Key code excerpt:

```diff
+from sglang.srt.layers.dp_attention import set_is_extend_in_batch
...
 forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
+set_is_extend_in_batch(batch.forward_mode.is_extend())
 _ = model_runner.forward(
     forward_batch=forward_batch, pp_proxy_tensors=pp_proxy
 )
```

- Reviewed files: `python/sglang/srt/managers/scheduler_pp_mixin.py`.
- Validation implications: GLM-5 pipeline-parallel profiling and dynamic chunking smoke tests should exercise extend-mode DSA batches, not just normal serving.

### PR #23540 - Split MI300X and MI325X options in GLM-5.1 generator

- Link: https://github.com/sgl-project/sglang/pull/23540
- State: merged at `2026-04-23T19:01:59Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `154` lines, `3` files.
- Motivation: the GLM-5.1 command generator previously collapsed MI300X and MI325X into a single selector item. That hid hardware-specific validation lanes and made AMD command generation less explicit.
- Key implementation: `glm-51-deployment.jsx` adds a separate `mi325x` hardware option, expands AMD checks to include `mi300x`, `mi325x`, and `mi355x`, and adds a dedicated `mi325x` BF16 TP/memory row. Docs navigation also moves GLM-5.1 to the front of the GLM group.
- Key code excerpts:

```diff
-{ id: 'mi300x', label: 'MI300X/MI325X', default: false },
+{ id: 'mi300x', label: 'MI300X',        default: false },
+{ id: 'mi325x', label: 'MI325X',        default: false },
```

```diff
-const isAMD = hw === 'mi300x' || hw === 'mi355x';
+const isAMD = ['mi300x', 'mi325x', 'mi355x'].includes(hw);
...
+mi325x: { bf16: { tp: 8, mem: 0.80 } },
```

- Reviewed files: `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`, `docs_new/docs.json`, `docs_new/cookbook/autoregressive/intro.mdx`.
- Validation implications: GLM-5.1 AMD command-generation tests should render MI300X, MI325X, and MI355X separately. Do not use one combined MI300X/MI325X row when recording perf or accuracy results.

## Cookbook Evidence

- sgl-cookbook PRs are documentation-parity inputs only until their diffs are reviewed with the same card standard. Do not cite them as runtime evidence without opening the cookbook diff.

## Validation Notes

- GLM-5/5.1 is a shared DSA/NSA lane. Any change to `deepseek_v2.py`, `deepseek_nextn.py`, `nsa_backend.py`, or `nsa_indexer.py` can affect DeepSeek V3.2 and GLM simultaneously.
- Preserve parser defaults in examples: `--tool-call-parser glm47` and `--reasoning-parser glm45`.
- Keep GLM-5 FP8, GLM-5 MXFP4, GLM-5 NVFP4, and GLM-5.1 FP8 validation separate; #22399 explicitly avoids pretending a GLM-5.1 NVFP4 checkpoint exists.
- For AMD, distinguish correctness CI from non-blocking performance CI and include `SGLANG_USE_AITER=1` lanes where the diff depends on AITER.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# SGLANG GLM-5 / GLM-5.1 PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2026-02-10 | [#18521](https://github.com/sgl-project/sglang/pull/18521) | merged | Support GlmMoeDsaForCausalLM | model wrapper, MoE/router, docs/config | `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/models/glm4_moe.py` |
| 2026-02-13 | [#18804](https://github.com/sgl-project/sglang/pull/18804) | merged | Fix GLM-5 fused shared expert | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |
| 2026-02-17 | [#18911](https://github.com/sgl-project/sglang/pull/18911) | merged | [AMD] [GLM-5 Day 0] Add GLM-5 nightly test | tests/benchmarks | `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `.github/workflows/nightly-test-amd-rocm720.yml` |
| 2026-03-06 | [#20062](https://github.com/sgl-project/sglang/pull/20062) | merged | [V32/GLM5] Control the threshold of applying dense attention with an environ | attention/backend, quantization, tests/benchmarks, docs/config | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py`, `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` |
| 2026-03-30 | [#21710](https://github.com/sgl-project/sglang/pull/21710) | merged | [AMD] Add GLM-5-FP8 nightly performance benchmarks for MI30x and MI35x | tests/benchmarks | `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `.github/workflows/nightly-test-amd-rocm720.yml` |
| 2026-03-31 | [#21773](https://github.com/sgl-project/sglang/pull/21773) | merged | [AMD][CI] Add GLM-5-MXFP4 accuracy and perf nightly tests for MI35x | quantization, tests/benchmarks | `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py`, `.github/workflows/nightly-test-amd.yml` |
| 2026-04-06 | [#22179](https://github.com/sgl-project/sglang/pull/22179) | merged | [Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation | docs/config | `docs/basic_usage/deepseek_v32.md` |
| 2026-04-07 | [#22285](https://github.com/sgl-project/sglang/pull/22285) | merged | Add CI tests for GLM-5 | model wrapper, tests/benchmarks | `test/registered/8-gpu-models/test_dsa_models_basic.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py` |
| 2026-04-08 | [#22314](https://github.com/sgl-project/sglang/pull/22314) | merged | [AMD] Fix GLM-5 fp8 KV quant path dispatch on MI300 | scheduler/runtime | `python/sglang/srt/mem_cache/memory_pool.py` |
| 2026-04-08 | [#22336](https://github.com/sgl-project/sglang/pull/22336) | merged | [AMD] Add GLM-5.1-FP8 nightly accuracy and performance benchmarks for MI30x and MI35x | tests/benchmarks | `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py` |
| 2026-04-08 | [#22399](https://github.com/sgl-project/sglang/pull/22399) | merged | [CI] Add GLM-5.1 nightly tests and update Qwen3.5 model | model wrapper, quantization, tests/benchmarks | `test/registered/8-gpu-models/test_glm_51_fp8.py`, `test/registered/8-gpu-models/test_qwen35.py`, `test/registered/gb300/test_glm5_fp8.py` |
| 2026-04-10 | [#22543](https://github.com/sgl-project/sglang/pull/22543) | merged | GLM-5/5.1 MXFP4 Checkpoint Inference Compatibility Fix | model wrapper | `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/server_args.py` |
| 2026-04-11 | [#22595](https://github.com/sgl-project/sglang/pull/22595) | merged | fix: normalize tool message content for GLM5.1 chat template | tests/benchmarks | `test/registered/openai_server/basic/test_serving_chat.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py` |
| 2026-04-13 | [#22712](https://github.com/sgl-project/sglang/pull/22712) | merged | [NPU] update glm5 running guide | docs/config | `docs/platforms/ascend/ascend_npu_glm5_examples.md` |
| 2026-04-15 | [#22850](https://github.com/sgl-project/sglang/pull/22850) | merged | [AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion) | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-04-17 | [#23060](https://github.com/sgl-project/sglang/pull/23060) | merged | [fix] Fix dynamic chunking profiling crash on GLM-5 models | scheduler/runtime | `python/sglang/srt/managers/scheduler_pp_mixin.py` |
| 2026-04-20 | [#23219](https://github.com/sgl-project/sglang/pull/23219) | merged | [AMD] Enable MTP for GLM-5-mxfp4 model | model wrapper | `python/sglang/srt/models/deepseek_nextn.py` |
| 2026-04-23 | [#23540](https://github.com/sgl-project/sglang/pull/23540) | merged | docs: split MI300X and MI325X options in GLM-5.1 generator | docs/config | `docs_new/docs.json`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`, `docs_new/cookbook/autoregressive/intro.mdx` |

## Diff Cards

### PR #18521 - Support GlmMoeDsaForCausalLM

- Link: https://github.com/sgl-project/sglang/pull/18521
- Status/date: `merged`, created 2026-02-10, merged 2026-02-10; author `JustinTong0323`.
- Diff scope read: `3` files, `+22/-7`; areas: model wrapper, MoE/router, docs/config; keywords: kv, moe, config, mla, spec, attention, cuda, eagle, flash, topk.
- Code diff details:
  - `python/sglang/srt/configs/model_config.py` modified +6/-5 (11 lines); hunks: def is_deepseek_nsa(config: PretrainedConfig) -> bool:; def from_server_args(; symbols: is_deepseek_nsa, from_server_args, _config_draft_model, _derive_model_shapes
  - `python/sglang/srt/server_args.py` modified +10/-1 (11 lines); hunks: def _handle_model_specific_adjustments(self):; def _handle_speculative_decoding(self):; symbols: _handle_model_specific_adjustments, _handle_speculative_decoding, _handle_deterministic_inference, auto_choose_speculative_params
  - `python/sglang/srt/models/glm4_moe.py` modified +6/-1 (7 lines); hunks: from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode; def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):; symbols: set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/models/glm4_moe.py`; keywords observed in patches: kv, moe, config, mla, spec, attention. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/models/glm4_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18804 - Fix GLM-5 fused shared expert

- Link: https://github.com/sgl-project/sglang/pull/18804
- Status/date: `merged`, created 2026-02-13, merged 2026-02-16; author `FrankMinions`.
- Diff scope read: `1` files, `+2/-1`; areas: model wrapper, MoE/router; keywords: eagle, expert, kv, moe.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +2/-1 (3 lines); hunks: def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):; symbols: set_eagle3_layers_to_capture, GlmMoeDsaForCausalLM, determine_num_fused_shared_experts
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe.py`; keywords observed in patches: eagle, expert, kv, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18911 - [AMD] [GLM-5 Day 0] Add GLM-5 nightly test

- Link: https://github.com/sgl-project/sglang/pull/18911
- Status/date: `merged`, created 2026-02-17, merged 2026-02-25; author `michaelzhang-ai`.
- Diff scope read: `5` files, `+635/-1`; areas: tests/benchmarks; keywords: test, attention, benchmark, cache, config, doc, moe.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py` added +249/-0 (249 lines); hunks: +"""MI35x GLM-5 GSM8K Completion Evaluation Test (8-GPU); symbols: ModelConfig:, get_display_name, get_one_example, get_few_shot_examples
  - `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` added +244/-0 (244 lines); hunks: +"""AMD GLM-5 GSM8K Completion Evaluation Test (8-GPU); symbols: ModelConfig:, get_display_name, get_one_example, get_few_shot_examples
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +71/-0 (71 lines); hunks: on:; on:
  - `.github/workflows/nightly-test-amd.yml` modified +70/-0 (70 lines); hunks: on:; jobs:
  - `test/registered/amd/accuracy/mi30x/test_gsm8k_eval_amd.py` modified +1/-1 (2 lines); hunks: "meta-llama/Llama-3.2-3B-Instruct": 0.55,
- Optimization/support interpretation: The concrete diff surface is `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `.github/workflows/nightly-test-amd-rocm720.yml`; keywords observed in patches: test, attention, benchmark, cache, config, doc. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py`, `.github/workflows/nightly-test-amd-rocm720.yml`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20062 - [V32/GLM5] Control the threshold of applying dense attention with an environ

- Link: https://github.com/sgl-project/sglang/pull/20062
- Status/date: `merged`, created 2026-03-06, merged 2026-03-09; author `Fridge003`.
- Diff scope read: `6` files, `+32/-59`; areas: attention/backend, quantization, tests/benchmarks, docs/config; keywords: kv, flash, mla, topk, cache, quant, attention, config, cuda, fp4.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-46 (49 lines); hunks: compute_cu_seqlens,; # Reuse this workspace buffer across all NSA backend instances; symbols: NSAFlashMLAMetadata:, __init__, init_forward_metadata_replay_cuda_graph_from_precomputed, set_nsa_prefill_impl
  - `python/sglang/srt/server_args.py` modified +26/-3 (29 lines); hunks: def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
  - `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +0/-4 (4 lines); hunks: def setUpClass(cls):; def setUpClass(cls):; symbols: setUpClass, setUpClass
  - `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +0/-4 (4 lines); hunks: def setUpClass(cls):; def setUpClass(cls):; symbols: setUpClass, setUpClass
  - `python/sglang/srt/environ.py` modified +1/-2 (3 lines); hunks: class Envs:; symbols: Envs:
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py`, `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`; keywords observed in patches: kv, flash, mla, topk, cache, quant. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py`, `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21710 - [AMD] Add GLM-5-FP8 nightly performance benchmarks for MI30x and MI35x

- Link: https://github.com/sgl-project/sglang/pull/21710
- Status/date: `merged`, created 2026-03-30, merged 2026-04-08; author `michaelzhang-ai`.
- Diff scope read: `6` files, `+345/-5`; areas: tests/benchmarks; keywords: test, fp8, attention, config, benchmark, cache, kv, mla, quant.
- Code diff details:
  - `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py` added +143/-0 (143 lines); hunks: +"""MI35x Nightly performance benchmark for GLM-5.; symbols: generate_simple_markdown_report, TestGLM5PerfMI35x, setUpClass, test_glm5_perf
  - `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py` added +140/-0 (140 lines); hunks: +"""Nightly performance benchmark for GLM-5 on MI30x.; symbols: generate_simple_markdown_report, TestNightlyGLM5Performance, setUpClass, test_bench_glm5
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +25/-1 (26 lines); hunks: jobs:; jobs:
  - `.github/workflows/nightly-test-amd.yml` modified +25/-0 (25 lines); hunks: jobs:; jobs:
  - `test/registered/amd/accuracy/mi30x/test_glm5_eval_amd.py` modified +6/-2 (8 lines); hunks: def get_display_name(self) -> str:; def get_display_name(self) -> str:; symbols: get_display_name, get_display_name
- Optimization/support interpretation: The concrete diff surface is `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `.github/workflows/nightly-test-amd-rocm720.yml`; keywords observed in patches: test, fp8, attention, config, benchmark, cache. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/amd/perf/mi35x/test_glm5_perf_mi35x.py`, `test/registered/amd/perf/mi30x/test_glm5_perf_amd.py`, `.github/workflows/nightly-test-amd-rocm720.yml`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21773 - [AMD][CI] Add GLM-5-MXFP4 accuracy and perf nightly tests for MI35x

- Link: https://github.com/sgl-project/sglang/pull/21773
- Status/date: `merged`, created 2026-03-31, merged 2026-04-15; author `michaelzhang-ai`.
- Diff scope read: `4` files, `+528/-130`; areas: quantization, tests/benchmarks; keywords: fp4, test, benchmark, cache, config, doc, moe, quant.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py` added +281/-0 (281 lines); hunks: +"""MI35x GLM-5-MXFP4 GSM8K Completion Evaluation Test (8-GPU); symbols: get_model_path, ModelConfig:, __post_init__, get_display_name
  - `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py` added +187/-0 (187 lines); hunks: +"""MI35x Nightly performance benchmark for GLM-5-MXFP4 model.; symbols: generate_simple_markdown_report, get_model_path, TestGLM5MXFP4PerfMI35x, setUpClass
  - `.github/workflows/nightly-test-amd.yml` modified +30/-66 (96 lines); hunks: on:; on:
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +30/-64 (94 lines); hunks: on:; on:
- Optimization/support interpretation: The concrete diff surface is `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py`, `.github/workflows/nightly-test-amd.yml`; keywords observed in patches: fp4, test, benchmark, cache, config, doc. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/amd/accuracy/mi35x/test_glm5_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_glm5_mxfp4_perf_mi35x.py`, `.github/workflows/nightly-test-amd.yml`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22179 - [Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation

- Link: https://github.com/sgl-project/sglang/pull/22179
- Status/date: `merged`, created 2026-04-06, merged 2026-04-06; author `mmangkad`.
- Diff scope read: `1` files, `+11/-12`; areas: docs/config; keywords: attention, benchmark, cache, config, deepep, doc, eagle, flash, fp8, kv.
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +11/-12 (23 lines); hunks: DeepSeek-V3.2 model family equips DeepSeek-V3.1-Terminus with DeepSeek Sparse Attention (DSA) through continued training. With DSA, a fine-grained sparse attent
- Optimization/support interpretation: The concrete diff surface is `docs/basic_usage/deepseek_v32.md`; keywords observed in patches: attention, benchmark, cache, config, deepep, doc. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/basic_usage/deepseek_v32.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22285 - Add CI tests for GLM-5

- Link: https://github.com/sgl-project/sglang/pull/22285
- Status/date: `merged`, created 2026-04-07, merged 2026-04-08; author `Fridge003`.
- Diff scope read: `2` files, `+153/-30`; areas: model wrapper, tests/benchmarks; keywords: attention, config, cuda, fp8, kv, test, eagle, spec.
- Code diff details:
  - `test/registered/8-gpu-models/test_dsa_models_basic.py` renamed +121/-1 (122 lines); hunks: write_github_step_summary,; def test_bs_1_speed(self):; symbols: TestDeepseekV32DP, test_bs_1_speed, TestGLM5DP, setUpClass
  - `test/registered/8-gpu-models/test_dsa_models_mtp.py` renamed +32/-29 (61 lines); hunks: register_cuda_ci(est_time=720, suite="stage-c-test-8-gpu-h200"); def setUpClass(cls):; symbols: TestDeepseekV32DPMTP, setUpClass, tearDownClass, test_bs_1_speed
- Optimization/support interpretation: The concrete diff surface is `test/registered/8-gpu-models/test_dsa_models_basic.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`; keywords observed in patches: attention, config, cuda, fp8, kv, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/8-gpu-models/test_dsa_models_basic.py`, `test/registered/8-gpu-models/test_dsa_models_mtp.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22314 - [AMD] Fix GLM-5 fp8 KV quant path dispatch on MI300

- Link: https://github.com/sgl-project/sglang/pull/22314
- Status/date: `merged`, created 2026-04-08, merged 2026-04-08; author `1am9trash`.
- Diff scope read: `1` files, `+27/-31`; areas: scheduler/runtime; keywords: attention, cache, fp8, kv, mla, quant, triton.
- Code diff details:
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +27/-31 (58 lines); hunks: quantize_k_cache,; def set_mla_kv_buffer(; symbols: set_mla_kv_buffer
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/mem_cache/memory_pool.py`; keywords observed in patches: attention, cache, fp8, kv, mla, quant. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/mem_cache/memory_pool.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22336 - [AMD] Add GLM-5.1-FP8 nightly accuracy and performance benchmarks for MI30x and MI35x

- Link: https://github.com/sgl-project/sglang/pull/22336
- Status/date: `merged`, created 2026-04-08, merged 2026-04-09; author `michaelzhang-ai`.
- Diff scope read: `6` files, `+918/-25`; areas: tests/benchmarks; keywords: test, fp8, attention, benchmark, cache, config, doc, fp4, kv, mla.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py` added +242/-0 (242 lines); hunks: +"""MI35x GLM-5.1 GSM8K Completion Evaluation Test (8-GPU); symbols: ModelConfig:, get_display_name, get_one_example, get_few_shot_examples
  - `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py` added +238/-0 (238 lines); hunks: +"""AMD GLM-5.1 GSM8K Completion Evaluation Test (8-GPU); symbols: ModelConfig:, get_display_name, get_one_example, get_few_shot_examples
  - `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py` added +146/-0 (146 lines); hunks: +"""MI35x Nightly performance benchmark for GLM-5.1.; symbols: generate_simple_markdown_report, TestGLM51PerfMI35x, setUpClass, test_glm51_perf
  - `test/registered/amd/perf/mi30x/test_glm51_perf_amd.py` added +138/-0 (138 lines); hunks: +"""Nightly performance benchmark for GLM-5.1 on MI30x.; symbols: generate_simple_markdown_report, TestNightlyGLM51Performance, setUpClass, test_bench_glm51
  - `.github/workflows/nightly-test-amd.yml` modified +87/-4 (91 lines); hunks: on:; on:
- Optimization/support interpretation: The concrete diff surface is `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py`; keywords observed in patches: test, fp8, attention, benchmark, cache, config. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_glm51_eval_amd.py`, `test/registered/amd/perf/mi35x/test_glm51_perf_mi35x.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22399 - [CI] Add GLM-5.1 nightly tests and update Qwen3.5 model

- Link: https://github.com/sgl-project/sglang/pull/22399
- Status/date: `merged`, created 2026-04-08, merged 2026-04-09; author `Kangyan-Zhou`.
- Diff scope read: `3` files, `+82/-6`; areas: model wrapper, quantization, tests/benchmarks; keywords: cuda, fp8, test, attention, eagle, spec, topk.
- Code diff details:
  - `test/registered/8-gpu-models/test_glm_51_fp8.py` added +69/-0 (69 lines); hunks: +import unittest; symbols: TestGlm51Fp8, test_glm51_fp8
  - `test/registered/8-gpu-models/test_qwen35.py` modified +10/-3 (13 lines); hunks: # Runs on both H200 and B200 via nightly-8-gpu-common suite; def test_qwen35(self):; symbols: TestQwen35, test_qwen35, test_qwen35
  - `test/registered/gb300/test_glm5_fp8.py` modified +3/-3 (6 lines); hunks: register_cuda_ci(est_time=7200, suite="nightly-4-gpu-gb300", nightly=True); class TestGlm5Fp8(unittest.TestCase):; symbols: TestGlm5Fp8, test_glm5_fp8, test_glm5_fp8
- Optimization/support interpretation: The concrete diff surface is `test/registered/8-gpu-models/test_glm_51_fp8.py`, `test/registered/8-gpu-models/test_qwen35.py`, `test/registered/gb300/test_glm5_fp8.py`; keywords observed in patches: cuda, fp8, test, attention, eagle, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/8-gpu-models/test_glm_51_fp8.py`, `test/registered/8-gpu-models/test_qwen35.py`, `test/registered/gb300/test_glm5_fp8.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22543 - GLM-5/5.1 MXFP4 Checkpoint Inference Compatibility Fix

- Link: https://github.com/sgl-project/sglang/pull/22543
- Status/date: `merged`, created 2026-04-10, merged 2026-04-14; author `ColinZ22`.
- Diff scope read: `3` files, `+8/-0`; areas: model wrapper; keywords: config, quant, cuda, fp4, kv, moe.
- Code diff details:
  - `python/sglang/srt/model_loader/loader.py` modified +3/-0 (3 lines); hunks: def _get_quantization_config(; symbols: _get_quantization_config
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +3/-0 (3 lines); hunks: def post_load_weights(; symbols: post_load_weights
  - `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: def _handle_missing_default_values(self):; symbols: _handle_missing_default_values
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/server_args.py`; keywords observed in patches: config, quant, cuda, fp4, kv, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/model_loader/loader.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22595 - fix: normalize tool message content for GLM5.1 chat template

- Link: https://github.com/sgl-project/sglang/pull/22595
- Status/date: `merged`, created 2026-04-11, merged 2026-04-16; author `whybeyoung`.
- Diff scope read: `2` files, `+67/-1`; areas: tests/benchmarks; keywords: cuda, doc, test.
- Code diff details:
  - `test/registered/openai_server/basic/test_serving_chat.py` modified +41/-1 (42 lines); hunks: ChatCompletionRequest,; def test_required_without_parser_invalid_json_returns_none(self):; symbols: test_required_without_parser_invalid_json_returns_none, TestNormalizeToolContent, test_openai_text_parts_flattened, test_multiple_text_parts_joined
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +26/-0 (26 lines); hunks: logger = logging.getLogger(__name__); def _apply_jinja_template(; symbols: normalize_tool_content, _extract_max_dynamic_patch, _apply_jinja_template
- Optimization/support interpretation: The concrete diff surface is `test/registered/openai_server/basic/test_serving_chat.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`; keywords observed in patches: cuda, doc, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/openai_server/basic/test_serving_chat.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22712 - [NPU] update glm5 running guide

- Link: https://github.com/sgl-project/sglang/pull/22712
- Status/date: `merged`, created 2026-04-13, merged 2026-04-13; author `zhsurpass`.
- Diff scope read: `1` files, `+8/-2`; areas: docs/config; keywords: doc.
- Code diff details:
  - `docs/platforms/ascend/ascend_npu_glm5_examples.md` modified +8/-2 (10 lines); hunks: docker run -itd --shm-size=16g --privileged=true --name ${NAME} \
- Optimization/support interpretation: The concrete diff surface is `docs/platforms/ascend/ascend_npu_glm5_examples.md`; keywords observed in patches: doc. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/platforms/ascend/ascend_npu_glm5_examples.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22850 - [AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion)

- Link: https://github.com/sgl-project/sglang/pull/22850
- Status/date: `merged`, created 2026-04-15, merged 2026-04-19; author `1am9trash`.
- Diff scope read: `1` files, `+24/-5`; areas: attention/backend; keywords: attention, cache, cuda, fp8, kv, quant.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +24/-5 (29 lines); hunks: from sglang.srt.environ import envs; _is_npu = is_npu(); symbols: __init__, _weights_proj_bf16_in_fp32_out, _store_index_k_cache
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; keywords observed in patches: attention, cache, cuda, fp8, kv, quant. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23060 - [fix] Fix dynamic chunking profiling crash on GLM-5 models

- Link: https://github.com/sgl-project/sglang/pull/23060
- Status/date: `merged`, created 2026-04-17, merged 2026-04-23; author `Baichuan7`.
- Diff scope read: `1` files, `+3/-0`; areas: scheduler/runtime; keywords: attention, scheduler.
- Code diff details:
  - `python/sglang/srt/managers/scheduler_pp_mixin.py` modified +3/-0 (3 lines); hunks: get_attention_dp_rank,; def profile_and_init_predictor(self: Scheduler):; symbols: profile_and_init_predictor
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/managers/scheduler_pp_mixin.py`; keywords observed in patches: attention, scheduler. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/managers/scheduler_pp_mixin.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23219 - [AMD] Enable MTP for GLM-5-mxfp4 model

- Link: https://github.com/sgl-project/sglang/pull/23219
- Status/date: `merged`, created 2026-04-20, merged 2026-04-20; author `1am9trash`.
- Diff scope read: `1` files, `+41/-15`; areas: model wrapper; keywords: attention, config, fp8, processor, quant, spec.
- Code diff details:
  - `python/sglang/srt/models/deepseek_nextn.py` modified +41/-15 (56 lines); hunks: is_dp_attention_enabled,; def __init__(; symbols: __init__, forward, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_nextn.py`; keywords observed in patches: attention, config, fp8, processor, quant, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_nextn.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23540 - docs: split MI300X and MI325X options in GLM-5.1 generator

- Link: https://github.com/sgl-project/sglang/pull/23540
- Status/date: `merged`, created 2026-04-23, merged 2026-04-23; author `zijiexia`.
- Diff scope read: `3` files, `+15/-13`; areas: docs/config; keywords: doc, flash, fp8, quant, spec.
- Code diff details:
  - `docs_new/docs.json` modified +8/-8 (16 lines); hunks: {
  - `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx` modified +6/-4 (10 lines); hunks: export const GLM51Deployment = () => {; export const GLM51Deployment = () => {
  - `docs_new/cookbook/autoregressive/intro.mdx` modified +1/-1 (2 lines); hunks: metatags:
- Optimization/support interpretation: The concrete diff surface is `docs_new/docs.json`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`, `docs_new/cookbook/autoregressive/intro.mdx`; keywords observed in patches: doc, flash, fp8, quant, spec. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs_new/docs.json`, `docs_new/src/snippets/autoregressive/glm-51-deployment.jsx`, `docs_new/cookbook/autoregressive/intro.mdx`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
