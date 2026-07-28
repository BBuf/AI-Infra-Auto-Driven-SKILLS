# Comprehensive AI Infrastructure Skills Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring every skill, executable, generated evidence artifact, operational reference, repository manifest, and validation record up to the verified July 27, 2026 state and publish the complete refresh as one draft pull request.

**Architecture:** Keep the existing twelve-skill layout and public CLIs. Repair correctness at the smallest testable function boundary, then regenerate evidence from immutable upstream commits and update the skill/reference layer to match those results. Finish with local, remote B200, profiler, Nsight Compute, plugin, and GitHub CI validation.

**Tech Stack:** Python 3.12, `unittest`, `pytest`, YAML/JSON/Markdown, Git/GitHub CLI, Claude plugin validators, pre-commit, SGLang/vLLM/TensorRT-LLM/TokenSpeed upstream Git histories, SGLang on NVIDIA B200, torch profiler, and Nsight Compute.

---

The task set below covers all twelve skills. A skill may remain behaviorally
unchanged only when its current implementation, references, and commands pass
the corresponding source and validation audits.

## File Responsibility Map

- `skills/llm-pipeline-analysis/scripts/`: trace boundary discovery, steady-state selection, layer/kernel reporting, and JSON export.
- `skills/llm-serving-capacity-planner/scripts/capacity_analyzer.py`: SGLang and vLLM startup-log parsing, memory decomposition, and concurrency reporting.
- `skills/model-compute-simulation/scripts/`: static operator simulation, trace extraction, measured timing, and MFU comparison.
- `skills/llm-torch-profiler-analysis/scripts/`: framework detection, live capture, trace triage, and candidate generation.
- `skills/llm-torch-profiler-analysis/references/`: current source paths and merged/open optimization evidence.
- `skills/sglang-prod-incident-triage/`: replay safety, current endpoints, and incident workflow.
- `tools/rebuild_model_pr_history_from_git.py`: deterministic SGLang/vLLM model-history discovery and rendering.
- `tools/check_open_pr_watch.py`: current open-PR collection with authenticated and anonymous REST paths.
- `model-pr-optimization-history/`: generated and manually curated bilingual PR evidence.
- `skills/model-architecture-diagram/`: public original-diagram source catalog and resolver.
- `skills/llm-serving-auto-benchmark/`: cross-framework recipe translation, validation, and current cookbook.
- Remaining `SKILL.md`, prompt, README, manifest, pre-commit, and workflow files: operational contract and packaging.
- `tests/`: behavior, schema, synchronization, and artifact-integrity regression coverage.

## Task 1: Repair Pipeline Analysis Semantics and JSON Output

**Files:**

- Modify: `tests/test_model_profiles.py`
- Modify: `skills/llm-pipeline-analysis/scripts/layer_timeline_analyzer.py`
- Modify: `skills/llm-pipeline-analysis/scripts/layer_kernel_breakdown.py`
- Modify: `skills/llm-pipeline-analysis/SKILL.md`

- [ ] **Step 1: Add failing tests for final/hash labeling, relative steady-state selection, TP=1 anchor fallback, top-k config fallback, and JSON comparison purity**

Add test cases with these assertions:

```python
def test_final_layer_label_wins_over_hash_suffix(self):
    timeline = load_timeline()
    breakdown = load_breakdown()
    self.assertEqual(
        timeline.layer_type_label(3, [0, 0, 128, 128], 4, 2),
        ("FINAL", 128),
    )
    self.assertEqual(breakdown._layer_type_label(128, 3, 4, 2), "FINAL")


def test_select_steady_state_pass_uses_relative_stability(self):
    timeline = load_timeline()
    self.assertEqual(
        timeline.select_steady_state_pass([1000.0, 5100.0, 5050.0, 5075.0]),
        1,
    )
    self.assertIsNone(
        timeline.select_steady_state_pass([1000.0, 2000.0, 4000.0, 8000.0])
    )


def test_generic_anchor_falls_back_to_repeated_rmsnorm(self):
    profiles = load_profiles()
    timeline = load_timeline()
    kernels = [{"name": "rms_norm_kernel"} for _ in range(8)]
    self.assertEqual(
        timeline.find_anchor_kernel(kernels, profiles.get_profile("generic")),
        "rms_norm",
    )


def test_moe_topk_accepts_num_experts_per_token_alias(self):
    breakdown = load_breakdown()
    self.assertEqual(
        breakdown.model_architecture_fields(
            {"moe": True, "num_experts": 128, "num_experts_per_token": 8}
        )["top_k"],
        8,
    )
```

Add a temporary trace/subprocess test that runs
`layer_kernel_breakdown.py --format json --compare-layer 1`, parses all stdout
with `json.loads`, and asserts that the result contains `primary`,
`comparison`, and `kernel_diff` objects with no prose before or after the JSON.

- [ ] **Step 2: Run the focused tests and confirm they fail for the audited reasons**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider \
  tests/test_model_profiles.py -q
```

Expected: failures show the current `HASH`/`FINAL` ordering, missing relative
steady-state helper, missing RMSNorm fallback, duplicate top-k lookup, and mixed
JSON/text comparison output.

- [ ] **Step 3: Implement relative steady-state selection and generic anchor fallback**

Add this independently testable helper and call it from the default CLI path:

```python
def select_steady_state_pass(
    layer_wall_us,
    *,
    relative_tolerance: float = 0.05,
    stable_pairs: int = 2,
):
    if stable_pairs < 1:
        raise ValueError("stable_pairs must be positive")
    streak = 0
    for index in range(1, len(layer_wall_us)):
        previous = float(layer_wall_us[index - 1])
        current = float(layer_wall_us[index])
        scale = max(abs(previous), abs(current), 1.0)
        if abs(current - previous) / scale <= relative_tolerance:
            streak += 1
            if streak >= stable_pairs:
                return index - stable_pairs
        else:
            streak = 0
    return None
```

Make `find_anchor_kernel` consider `rms_norm`, `rmsnorm`, and
`RMSNorm` after model-specific anchors and before `AllReduce`, so a TP=1 trace
does not require NCCL. Keep `--anchor-kernel` as the authoritative override.

- [ ] **Step 4: Unify layer labels, architecture fields, and JSON comparison**

Check `FINAL` before the hash suffix in `layer_type_label`. Extract a
`model_architecture_fields(config)` helper that reads:

```python
top_k = config.get(
    "num_experts_per_tok",
    config.get("num_experts_per_token", "?"),
)
```

Build JSON comparison as one document:

```python
payload = {
    "primary": json.loads(format_json_output(primary_kernels, ...)),
    "comparison": json.loads(format_json_output(comparison_kernels, ...)),
    "kernel_diff": {
        "only_primary": sorted(primary_names - comparison_names),
        "only_comparison": sorted(comparison_names - primary_names),
        "common_count": len(primary_names & comparison_names),
    },
}
print(json.dumps(payload, indent=2))
```

Do not call `print_layer_breakdown` or `print_diff` in JSON mode.

- [ ] **Step 5: Update the pipeline skill contract and run tests**

Document relative steady-state detection, the TP=1 RMSNorm fallback, and the
single-document JSON comparison contract. Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider \
  tests/test_model_profiles.py -q
```

Expected: all pipeline/profile tests pass.

- [ ] **Step 6: Commit the pipeline unit**

```bash
git add tests/test_model_profiles.py \
  skills/llm-pipeline-analysis/SKILL.md \
  skills/llm-pipeline-analysis/scripts/layer_timeline_analyzer.py \
  skills/llm-pipeline-analysis/scripts/layer_kernel_breakdown.py
git commit -m "Fix pipeline trace analysis semantics"
```

## Task 2: Add Real vLLM Capacity Evidence

**Files:**

- Modify: `tests/test_capacity_analyzer.py`
- Modify: `skills/llm-serving-capacity-planner/scripts/capacity_analyzer.py`
- Modify: `skills/llm-serving-capacity-planner/references/log-patterns.md`
- Modify: `skills/llm-serving-capacity-planner/SKILL.md`

- [ ] **Step 1: Add a failing current-vLLM startup-log fixture**

Use the exact current log shapes emitted by
`vllm/v1/worker/gpu_worker.py`,
`vllm/v1/worker/gpu_model_runner.py`, and
`vllm/v1/core/kv_cache_utils.py`:

```python
VLLM_LOG = """
INFO vllm engine starting
Initial free memory: 79.20 GiB; Requested memory: 0.900000 (util), 72.00 GiB
Model loading took 14.50 GiB memory and 12.000000 seconds
Available KV cache memory: 52.25 GiB
Graph capturing finished in 8 secs, took 1.25 GiB
GPU KV cache size: 1,572,864 tokens
Maximum concurrency for 8,192 tokens per request: 192.00x
"""


def test_parse_current_vllm_capacity_evidence(self):
    parsed = MOD.parse_log(VLLM_LOG)
    self.assertEqual(parsed.framework, "vllm")
    self.assertEqual(parsed.vllm.initial_free_gib, 79.20)
    self.assertEqual(parsed.vllm.model_loading_gib, 14.50)
    self.assertEqual(parsed.vllm.available_kv_cache_gib, 52.25)
    self.assertEqual(parsed.vllm.gpu_kv_cache_tokens, 1_572_864)
    self.assertEqual(parsed.vllm.max_model_len, 8192)
    self.assertEqual(parsed.vllm.maximum_concurrency, 192.0)
```

Also assert that JSON output includes these fields and does not invent SGLang
`mem_fraction_static`, pool checkpoints, or CUDA-graph batch size.

- [ ] **Step 2: Run the capacity tests and confirm the vLLM fixture fails**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider \
  tests/test_capacity_analyzer.py -q
```

Expected: existing SGLang tests pass and the new vLLM evidence fields are
missing.

- [ ] **Step 3: Implement a framework-specific vLLM evidence object and parser**

Add:

```python
@dataclass
class VllmMemoryInfo:
    initial_free_gib: Optional[float] = None
    requested_utilization: Optional[float] = None
    requested_memory_gib: Optional[float] = None
    model_loading_gib: Optional[float] = None
    available_kv_cache_gib: Optional[float] = None
    cuda_graph_gib: Optional[float] = None
    gpu_kv_cache_tokens: Optional[int] = None
    max_model_len: Optional[int] = None
    maximum_concurrency: Optional[float] = None
```

Add `vllm: VllmMemoryInfo = field(default_factory=VllmMemoryInfo)` to
`ParsedLog`. Parse comma-formatted token counts with `replace(",", "")`.
Populate only fields proven by a matching line.

- [ ] **Step 4: Render vLLM-specific text and JSON without fabricated fields**

For vLLM logs, render a `vLLM startup evidence` section containing initial
free/requested memory, model load, CUDA graphs, KV GiB, KV tokens, model length,
and reported theoretical concurrency. Use `null` in JSON for absent fields.
Keep SGLang memory decomposition unchanged.

- [ ] **Step 5: Remove dead parsing code and update references**

Remove the inert `for line_text in []` loop. Add the current vLLM source paths
and sample lines to `log-patterns.md`. State in `SKILL.md` that vLLM exposes a
different set of checkpoints and unknown values remain unknown.

- [ ] **Step 6: Run tests and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider \
  tests/test_capacity_analyzer.py -q
git add tests/test_capacity_analyzer.py \
  skills/llm-serving-capacity-planner
git commit -m "Parse current vLLM capacity logs"
```

Expected: all capacity tests pass.

## Task 3: Correct Measured MFU and Trace/Template Comparison

**Files:**

- Modify: `tests/test_model_profiles.py`
- Modify: `skills/model-compute-simulation/scripts/model_compute_simulator.py`
- Modify: `skills/model-compute-simulation/scripts/extract_compute_flow_from_trace.py`
- Modify: `skills/model-compute-simulation/SKILL.md`

- [ ] **Step 1: Add failing tests for kernel-flow timing and semantic op families**

Add a simulator test that loads:

```python
kernel_flow = {
    "metadata": {"total_dur_us": 2000, "compress_ratio": 0},
    "category_summary": {"gemm_bf16": {"dur_us": 2000, "count": 1}},
    "kernels": [
        {
            "name": "gemm",
            "simplified_name": "gemm",
            "dur_us": 2000,
            "category": "gemm_bf16",
        }
    ],
}
```

Run the CLI with `--kernel-flow` and assert:

- `measured_ms == 2.0 * num_hidden_layers`;
- `mfu_pct` is non-null and derives from that measured time;
- `--format json` produces one parseable JSON document containing
  `kernel_flow.metadata`.

Add direct tests for:

```python
self.assertEqual(mod.canonical_trace_op_family("aten::mm", "attention"), "matmul")
self.assertEqual(mod.canonical_template_op_family("q_proj", "attention"), "matmul")
self.assertEqual(mod.canonical_trace_op_family("aten::rms_norm", "norm"), "norm")
self.assertEqual(mod.canonical_template_op_family("rmsnorm", "norm"), "norm")
```

Assert that `min_flops=1` excludes a zero-FLOP record.

- [ ] **Step 2: Run focused tests and confirm failure**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider \
  tests/test_model_profiles.py -q
```

Expected: kernel-flow timing is applied after simulation, JSON is contaminated
by the text table, raw `aten::*` names cannot match semantic template names,
and zero-FLOP records survive a positive filter.

- [ ] **Step 3: Parse all measured timing inputs before `simulate()`**

Add a shared loader:

```python
def load_json_argument(value: str) -> dict:
    if value.startswith("@"):
        return load_json(value[1:])
    return json.loads(value)


def measured_ms_from_kernel_detail(detail: dict, num_layers: int) -> float:
    layer_us = float(detail.get("metadata", {}).get("total_dur_us", 0))
    if layer_us <= 0:
        raise ValueError("kernel detail metadata.total_dur_us must be positive")
    return layer_us / 1000.0 * num_layers
```

Resolve `--kernel-detail`, `--kernel-flow`, `--kernel-ms`, and
`--per-layer-ms` before calling `simulate()`. Preserve explicit
`--measured-ms` unless a more specific per-kernel option was selected.

- [ ] **Step 4: Make JSON mode contain kernel-flow data without appended prose**

Add `kernel_flow: Optional[dict] = None` to `SimResult`, include it in
`format_json`, and print `format_kernel_flow` only in text mode. Reject
nonpositive measured duration with an argparse error.

- [ ] **Step 5: Compare canonical operator families**

Introduce `canonical_trace_op_family` and
`canonical_template_op_family`. Use `matmul`, `attention`, `norm`, `router`,
`activation`, `embedding`, and `other` families. Compute missing/extra sets
from canonical families within each category, not raw `aten::*` versus
template operation names.

Change the positive FLOP filter to:

```python
if min_flops > 0:
    results = [row for row in results if row["flops"] >= min_flops]
```

- [ ] **Step 6: Update the skill, run tests, and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider \
  tests/test_model_profiles.py -q
git add tests/test_model_profiles.py skills/model-compute-simulation
git commit -m "Fix measured MFU and trace comparison"
```

Expected: all compute/profile tests pass and JSON output is independently
parseable.

## Task 4: Harden Trusted Incident Replay and Current HiCache Operations

**Files:**

- Create: `tests/test_incident_replay.py`
- Modify: `skills/sglang-prod-incident-triage/scripts/replay_trusted_request_dump.py`
- Modify: `skills/sglang-prod-incident-triage/references/endpoints-and-signals.md`
- Modify: `skills/sglang-prod-incident-triage/references/decision-tree.md`
- Modify: `skills/sglang-prod-incident-triage/SKILL.md`

- [ ] **Step 1: Add failing replay validation and HTTP-error tests**

Load the script with `importlib` and assert:

```python
def test_validate_replay_args_rejects_nonpositive_values(self):
    for values in (
        SimpleNamespace(speed=0, parallel=1, timeout=1),
        SimpleNamespace(speed=-1, parallel=1, timeout=1),
        SimpleNamespace(speed=1, parallel=0, timeout=1),
        SimpleNamespace(speed=1, parallel=1, timeout=0),
    ):
        with self.assertRaises(ValueError):
            MOD.validate_replay_args(values)


def test_run_one_request_raises_before_decoding_http_error(self):
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError("500")
    with patch.object(MOD.requests, "post", return_value=response):
        with self.assertRaises(requests.HTTPError):
            MOD.run_one_request(RECORD, ARGS, 0.0, 0.0, 0)
    response.json.assert_not_called()
```

- [ ] **Step 2: Run the new test and confirm failure**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider \
  tests/test_incident_replay.py -q
```

- [ ] **Step 3: Validate CLI values and HTTP status**

Add:

```python
def validate_replay_args(args: argparse.Namespace) -> None:
    if args.speed <= 0:
        raise ValueError("--speed must be greater than zero")
    if args.parallel <= 0:
        raise ValueError("--parallel must be greater than zero")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")
```

Call it immediately after parsing, routing errors through `parser.error`.
Call `response.raise_for_status()` before consuming streaming lines or JSON.

- [ ] **Step 4: Add the current HiCache clear endpoint to operational guidance**

Document:

```text
POST /hicache/storage-backend/clear
```

Keep attach and detach endpoints, explain that clear removes backend contents
without changing attachment state, and require incident artifacts before any
destructive cache operation.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider \
  tests/test_incident_replay.py -q
git add tests/test_incident_replay.py skills/sglang-prod-incident-triage
git commit -m "Harden incident replay and cache guidance"
```

## Task 5: Refresh Profiler Framework Detection, Source Paths, and PR States

**Files:**

- Modify: `tests/test_llm_torch_profiler_analysis.py`
- Modify: `skills/llm-torch-profiler-analysis/scripts/profile_common.py`
- Modify: `skills/llm-torch-profiler-analysis/scripts/triage_kernel_helpers.py`
- Modify: `skills/llm-torch-profiler-analysis/scripts/triage_overlap_helpers.py`
- Modify: `skills/llm-torch-profiler-analysis/references/source-map.md`
- Modify: `skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md`
- Modify: `skills/llm-torch-profiler-analysis/references/overlap-catalog.md`
- Modify: `skills/llm-torch-profiler-analysis/references/vllm-torch-compile-fusions.md`
- Modify: `skills/llm-torch-profiler-analysis/references/heuristics.md`
- Modify: `skills/llm-torch-profiler-analysis/SKILL.md`

- [ ] **Step 1: Add failing tests for unknown-framework handling and migrated SGLang paths**

Add:

```python
def test_auto_framework_does_not_guess_sglang(self):
    with self.assertRaisesRegex(ValueError, "--framework"):
        profile_common.resolve_framework(
            "auto",
            input_path=Path("/tmp/framework-neutral-trace"),
        )


def test_current_sglang_kernel_paths_replace_jit_kernel(self):
    text = "\n".join(
        [
            SOURCE_MAP.read_text(),
            FUSE_CATALOG.read_text(),
            inspect.getsource(kernel_helpers),
        ]
    )
    self.assertNotIn("python/sglang/jit_kernel/", text)
    self.assertNotIn("scheduler_profiler_mixin.py", text)
```

Add status assertions that open entries are exactly open and merged entries
are described as mainline/provenance, not `in-flight`.

- [ ] **Step 2: Run the profiler tests and confirm failure**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider \
  tests/test_llm_torch_profiler_analysis.py -q
```

- [ ] **Step 3: Stop silently defaulting unknown traces to SGLang**

Change the final `resolve_framework` branch to:

```python
raise ValueError(
    "Could not infer the serving framework; pass "
    "--framework sglang|vllm|trtllm|tokenspeed."
)
```

Preserve explicit framework selection and existing detection signatures.

- [ ] **Step 4: Apply the current SGLang path migration consistently**

Use the current source tree at
`sgl-project/sglang@8d6549bc4039d33635844495d86684677a4f0df8`
and replace stale path families:

```text
python/sglang/jit_kernel/csrc/...        -> python/sglang/kernels/jit/csrc/...
python/sglang/jit_kernel/diffusion/...   -> python/sglang/kernels/ops/diffusion/...
python/sglang/jit_kernel/fused_qknorm_rope.py
                                         -> python/sglang/kernels/ops/attention/fused_qknorm_rope.py
python/sglang/jit_kernel/fused_metadata_copy.py
                                         -> python/sglang/kernels/ops/attention/fused_metadata_copy.py
python/sglang/jit_kernel/fused_store_index_cache.py
                                         -> python/sglang/kernels/ops/attention/fused_store_index_cache.py
python/sglang/jit_kernel/norm.py          -> python/sglang/kernels/ops/layernorm/norm.py
python/sglang/jit_kernel/rope.py          -> python/sglang/kernels/ops/attention/rope.py
python/sglang/srt/managers/scheduler_profiler_mixin.py
                                         -> python/sglang/srt/managers/scheduler_components/profiler_manager.py
docs/developer_guide/benchmark_and_profiling.md
                                         -> docs_new/docs/developer_guide/benchmark_and_profiling.mdx
```

Resolve FLA, Mamba, MoE router, diffusion, and model-specific paths by
`git ls-files`; remove a reference if no current source or historical label
justifies it.

- [ ] **Step 5: Reclassify audited PR evidence**

Keep these SGLang PRs open: `#21877`, `#21889`, `#21491`, `#22005`,
`#20667`, and `#24168`. Move merged SGLang `#18612`, `#22918`, `#22851`,
`#24125`, `#24007`, and `#23965` to mainline provenance. Remove or explicitly
label closed-unmerged `#22392`, `#24150`, and `#21878`.

Apply the same rule to:

- FlashInfer `#2720` (open);
- TensorRT-LLM `#12525`, `#12544`, `#12738` (merged) and `#12557`
  (closed-unmerged);
- vLLM `#38445`, `#37646`, `#41263`, `#41428`, `#41255`, `#36823`
  (merged); `#38621`, `#36413`, `#41446` (open); and the audited
  closed-unmerged entries.

Do not imply that SGLang `#22392` landed; describe the current
`sgl_kernel.fp8_scaled_mm` mainline implementation from its actual source.

- [ ] **Step 6: Verify every current-path claim against the upstream checkout**

Extract repository-relative paths from the references and helpers, then run:

```bash
while IFS= read -r path; do
  test -e "/tmp/aiinfra-upstreams-20260727/sglang/$path" ||
    printf 'missing %s\n' "$path"
done < /tmp/sglang-profiler-paths.txt
```

Expected: no unmarked current SGLang path is missing. Historical PR-only paths
must carry an explicit historical/open-PR label.

- [ ] **Step 7: Run tests and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider \
  tests/test_llm_torch_profiler_analysis.py -q
git add tests/test_llm_torch_profiler_analysis.py \
  skills/llm-torch-profiler-analysis
git commit -m "Refresh profiler sources and evidence"
```

## Task 6: Complete Model-History Generation Coverage

**Files:**

- Create: `tests/test_model_pr_history_generator.py`
- Modify: `tools/rebuild_model_pr_history_from_git.py`
- Modify: `model-pr-optimization-history/SKILL.md`
- Modify: `model-pr-optimization-history/README.md`

- [ ] **Step 1: Add failing generator-consistency tests**

Load the generator module and assert:

```python
def test_framework_orders_cover_supported_current_models(self):
    self.assertTrue(
        {"hunyuan3-preview", "moss-vl", "qwen36"}
        <= set(MOD.FRAMEWORK_MODEL_ORDER["sglang"])
    )
    self.assertTrue(
        {"hunyuan3-preview", "qwen36"}
        <= set(MOD.FRAMEWORK_MODEL_ORDER["vllm"])
    )
    self.assertNotIn("moss-vl", MOD.FRAMEWORK_MODEL_ORDER["vllm"])


def test_every_framework_model_has_title_filter_and_subject_hints(self):
    for framework, models in MOD.FRAMEWORK_MODEL_ORDER.items():
        for model in models:
            self.assertIn(model, MOD.MODEL_TITLES)
            self.assertIn(model, MOD.MODEL_FILTERS[framework])
            self.assertIn(model, MOD.SUBJECT_HINTS)
```

Use synthetic file lists to prove Qwen3.6 docs/tests and Hunyuan3/MOSS-VL
runtime files are selected without contaminating Qwen3 core or Hunyuan3D.

- [ ] **Step 2: Run the new generator test and confirm the missing order entries**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider \
  tests/test_model_pr_history_generator.py -q
```

- [ ] **Step 3: Synchronize model order and subject filters**

Add SGLang Hunyuan3 Preview, MOSS-VL, and Qwen3.6; add vLLM Hunyuan3 Preview
and Qwen3.6. Keep vLLM MOSS-VL excluded because current `main` has no matching
implementation surface. Tighten Qwen3.6 hints to `qwen3.6`, `qwen36`,
`qwen3_6`, and exact current Qwen3.6 docs/tests.

Change the MiniMax title to `MiniMax M2/M3 Series`.

- [ ] **Step 4: Run generator tests and a dry run against immutable checkouts**

```bash
SGLANG_PR_HISTORY_ROOT=/tmp/aiinfra-upstreams-20260727/sglang \
VLLM_PR_HISTORY_ROOT=/tmp/aiinfra-upstreams-20260727/vllm \
MODEL_PR_HISTORY_CACHE=/tmp/model_pr_history_git_trace_cache_v6.json \
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/rebuild_model_pr_history_from_git.py --dry-run
```

Expected: 34 SGLang model histories and 33 vLLM model histories are selected;
the new five model/framework pairs contain real files and merged PR evidence.

- [ ] **Step 5: Update the lookup skill and commit the generator**

Add the five new framework/slug combinations to the skill's supported lookup
list and document that model availability is framework-specific.

```bash
git add tests/test_model_pr_history_generator.py \
  tools/rebuild_model_pr_history_from_git.py \
  model-pr-optimization-history/SKILL.md \
  model-pr-optimization-history/README.md
git commit -m "Complete model history generation coverage"
```

## Task 7: Regenerate SGLang/vLLM Histories and Manually Refresh TensorRT-LLM/TokenSpeed

**Files:**

- Modify: all generated `model-pr-optimization-history/sglang/*/README.{en,zh}.md`
- Modify: all generated `model-pr-optimization-history/vllm/*/README.{en,zh}.md`
- Create: `model-pr-optimization-history/sglang/hunyuan3-preview/README.en.md`
- Create: `model-pr-optimization-history/sglang/hunyuan3-preview/README.zh.md`
- Create: `model-pr-optimization-history/sglang/moss-vl/README.en.md`
- Create: `model-pr-optimization-history/sglang/moss-vl/README.zh.md`
- Create: `model-pr-optimization-history/sglang/qwen36/README.en.md`
- Create: `model-pr-optimization-history/sglang/qwen36/README.zh.md`
- Create: `model-pr-optimization-history/vllm/hunyuan3-preview/README.en.md`
- Create: `model-pr-optimization-history/vllm/hunyuan3-preview/README.zh.md`
- Create: `model-pr-optimization-history/vllm/qwen36/README.en.md`
- Create: `model-pr-optimization-history/vllm/qwen36/README.zh.md`
- Modify: `model-pr-optimization-history/{sglang,vllm}/README.md`
- Modify: `model-pr-optimization-history/tensorrt_llm/{README.md,kimi/README.en.md,kimi/README.zh.md,qwen35/README.en.md,qwen35/README.zh.md}`
- Modify: `model-pr-optimization-history/tokenspeed/{README.md,kimi/README.en.md,kimi/README.zh.md,qwen35/README.en.md,qwen35/README.zh.md}`

- [ ] **Step 1: Reconfirm and fast-forward the four source heads**

```bash
git -C /tmp/aiinfra-upstreams-20260727/sglang fetch origin main
git -C /tmp/aiinfra-upstreams-20260727/sglang merge --ff-only origin/main
git -C /tmp/aiinfra-upstreams-20260727/vllm fetch origin main
git -C /tmp/aiinfra-upstreams-20260727/vllm merge --ff-only origin/main
git -C /tmp/aiinfra-upstreams-20260727/TensorRT-LLM fetch origin main
git -C /tmp/aiinfra-upstreams-20260727/TensorRT-LLM merge --ff-only origin/main
git -C /tmp/aiinfra-upstreams-20260727/tokenspeed fetch origin main
git -C /tmp/aiinfra-upstreams-20260727/tokenspeed merge --ff-only origin/main
```

Record the final full SHAs immediately before generation.

- [ ] **Step 2: Regenerate SGLang and vLLM histories**

```bash
SGLANG_PR_HISTORY_ROOT=/tmp/aiinfra-upstreams-20260727/sglang \
VLLM_PR_HISTORY_ROOT=/tmp/aiinfra-upstreams-20260727/vllm \
MODEL_PR_HISTORY_CACHE=/tmp/model_pr_history_git_trace_cache_v6.json \
python3 tools/rebuild_model_pr_history_from_git.py
```

Expected: bilingual files and indexes carry the same current full SHA and
refresh date; no history is empty because of a configuration omission.

- [ ] **Step 3: Manually diff-review TensorRT-LLM and TokenSpeed changes since the prior heads**

Use:

```bash
git -C /tmp/aiinfra-upstreams-20260727/TensorRT-LLM log \
  --oneline aaffa2f9fef3025e0f698d978385a73460344e0b..HEAD -- \
  'tensorrt_llm/**/*kimi*' 'tensorrt_llm/**/*qwen3_5*' 'tests/**/*kimi*' 'tests/**/*qwen3_5*'

git -C /tmp/aiinfra-upstreams-20260727/tokenspeed log \
  --oneline d0a7faddb5ec0d4c6d037c4c3e6a781d2c5164a8..HEAD -- \
  '*kimi*' '*qwen3_5*'
```

For each candidate merged PR, read its full file list and patch with
`gh pr diff`. Add a bilingual card only when the diff is genuinely relevant.
Update the source-head section even when no new card qualifies.

- [ ] **Step 4: Run dossier quality and stale-head checks**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider \
  tests/test_model_pr_dossier_quality.py \
  tests/test_model_pr_history_generator.py -q

rg -n '2026-06-27 Source Head Refresh|2026-06-27 源码 head 刷新|\
aaffa2f9fef3025e0f698d978385a73460344e0b|\
d0a7faddb5ec0d4c6d037c4c3e6a781d2c5164a8' \
  model-pr-optimization-history
```

Expected: tests pass and the stale scan has no current-head claims left; old
SHAs may remain only in explicitly historical provenance paragraphs.

- [ ] **Step 5: Commit generated and curated histories**

```bash
git add model-pr-optimization-history
git commit -m "Refresh model optimization histories"
```

## Task 8: Refresh the Open PR Watch Without Empty-Report Failure

**Files:**

- Modify: `tools/check_open_pr_watch.py`
- Modify: `tests/test_open_pr_watch.py`
- Modify: `model-pr-optimization-history/open-pr-watch.md`

- [ ] **Step 1: Extend watch vocabulary with current families**

Add `MiniMax M3`, `Kimi K3`, `Inkling`, and `Unlimited OCR` to
`DEFAULT_TERMS`. Add tests that matching is case-insensitive and the new terms
are rendered once per PR even if several terms match.

- [ ] **Step 2: Run tests, implement, and regenerate**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider \
  tests/test_open_pr_watch.py -q
python3 tools/check_open_pr_watch.py --format markdown \
  --output model-pr-optimization-history/open-pr-watch.md
python3 tools/check_open_pr_watch.py --format json \
  --output /tmp/open-pr-watch-20260727.json
```

Expected: the committed Markdown is dated July 27, 2026, the JSON is nonempty,
all four repositories succeeded through `gh` or REST fallback, and every
listed PR is open when checked via GitHub.

- [ ] **Step 3: Commit**

```bash
git add tools/check_open_pr_watch.py tests/test_open_pr_watch.py \
  model-pr-optimization-history/open-pr-watch.md
git commit -m "Refresh upstream open PR watch"
```

## Task 10: Add Kimi K3 to the Public Architecture Catalog

**Files:**

- Modify: `tests/test_model_architecture_diagram.py`
- Modify: `skills/model-architecture-diagram/references/diagram-index.json`
- Modify: `skills/model-architecture-diagram/references/source-notes.md`
- Modify: `skills/model-architecture-diagram/SKILL.md`

- [ ] **Step 1: Add a failing Kimi K3 resolution test**

```python
def test_kimi_k3_resolves_public_original_diagram(self):
    result = self.mod.resolve("moonshotai/kimi-k3")
    self.assertEqual(result[0]["id"], "kimi-k3-architecture")
    self.assertEqual(result[0]["source"], "InfraTech")
    self.assertTrue(result[0]["url"].endswith("/models/kimi_k_3/kimi_k_3_architecture.jpg"))
```

- [ ] **Step 2: Add the source-linked catalog entry**

Record source commit
`CalvinXKY/InfraTech@16a34a7494f0f7b270501064033e3ce35ef41bdf`
and add aliases `kimi-k3`, `kimi k3`, `kimi_k3`, `moonshotai/kimi-k3`.
Use the raw public original URL; do not vendor the 10,850 × 12,619 image or
replace historical release archives.

- [ ] **Step 3: Run tests and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider \
  tests/test_model_architecture_diagram.py -q
git add tests/test_model_architecture_diagram.py \
  skills/model-architecture-diagram
git commit -m "Add Kimi K3 architecture source"
```

## Task 11: Refresh Serving Cookbook Flags and Current Model Recipes

**Files:**

- Modify: `tests/test_llm_serving_cookbook_configs.py`
- Modify: `tests/test_llm_serving_docs.py`
- Modify: `skills/llm-serving-auto-benchmark/scripts/validate_cookbook_configs.py`
- Modify: all `skills/llm-serving-auto-benchmark/configs/cookbook-llm/*.yaml`
- Create: `skills/llm-serving-auto-benchmark/configs/cookbook-llm/minimax-m3.yaml`
- Create: `skills/llm-serving-auto-benchmark/configs/cookbook-llm/qwen36-35b-a3b-fp8.yaml`
- Modify: `skills/llm-serving-auto-benchmark/configs/cookbook-llm/README.md`
- Modify: `skills/llm-serving-auto-benchmark/references/example-plan.yaml`
- Modify: `skills/llm-serving-auto-benchmark/references/framework-reference.md`
- Modify: `skills/llm-serving-auto-benchmark/SKILL.md`

- [ ] **Step 1: Add failing retired-flag and new-recipe tests**

Assert that no config, example, static flag set, or skill text contains:

```text
max_num_partial_prefills
max_long_partial_prefills
```

Assert that `long_prefill_token_threshold` remains supported, and that both
new configs pass schema validation and render at least one enabled-framework
command.

- [ ] **Step 2: Remove retired vLLM flags everywhere**

Remove the two keys from all YAML search spaces,
`STATIC_SERVER_FLAGS["vllm"]`, examples, and prose. Do not remove
`long_prefill_token_threshold`, which remains in current vLLM.

- [ ] **Step 3: Add authoritative current recipes**

Add:

- `MiniMaxAI/MiniMax-M3-MXFP8` for B200 and the official MiniMax M3 launch
  constraints represented by `minimax-m3.yaml`;
- `Qwen/Qwen3.6-35B-A3B-FP8` represented by
  `qwen36-35b-a3b-fp8.yaml`.

Enable a framework only after its current checkout/help exposes the model and
flags. For an unsupported framework, use:

```yaml
framework_name:
  enabled: false
  support_status: not_verified_at_recorded_head
```

Do not add Inkling, Unlimited OCR, Kimi K3, or DeepSeek V4 to the
cross-framework cookbook in this refresh: their current endpoint, checkpoint,
or four-framework comparison contract is not sufficiently uniform. Mention
that exclusion in the cookbook README.

- [ ] **Step 4: Validate static and live CLI flag surfaces**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider \
  tests/test_llm_serving_cookbook_configs.py \
  tests/test_llm_serving_docs.py -q

python3 skills/llm-serving-auto-benchmark/scripts/validate_cookbook_configs.py \
  --print-commands \
  skills/llm-serving-auto-benchmark/configs/cookbook-llm/*.yaml
```

Expected: every YAML is valid, bounded, renders commands, and contains no
retired flag.

- [ ] **Step 5: Commit**

```bash
git add tests/test_llm_serving_cookbook_configs.py \
  tests/test_llm_serving_docs.py \
  skills/llm-serving-auto-benchmark
git commit -m "Refresh serving cookbook recipes"
```

## Task 12: Synchronize Remaining Skills, Model Profiles, and SOTA Prompts

**Files:**

- Modify: `skills/model-compute-simulation/references/model-config-index.json`
- Modify: `skills/model-optimization/model-pr-diff-dossier/SKILL.md`
- Modify: `skills/model-optimization/model-pr-diff-dossier/references/card-schema.md`
- Modify: `skills/sglang-sota-humanize-loop/SKILL.md`
- Modify: `skills/sglang-sota-humanize-loop/references/refined-plan-template.md`
- Modify: `skills/vllm-sota-humanize-loop/SKILL.md`
- Modify: `skills/vllm-sota-humanize-loop/references/refined-plan-template.md`
- Modify: `prompts/sglang-sota-b200-prompts.md`
- Modify: `prompts/sglang-sota-b200-codex-goal-prompts.md`
- Modify: `prompts/sglang-sota-h200-prompts.md`
- Modify: `prompts/sglang-sota-h200-codex-goal-prompts.md`
- Modify: `tests/test_model_profiles.py`
- Modify: `tests/test_model_pr_dossier_quality.py`
- Modify: `tests/test_sota_humanize_loop_docs.py`
- Modify: `tests/test_sglang_sota_humanize_loop_docs.py`

- [ ] **Step 1: Add synchronization tests before editing shared guidance**

Assert that:

- the four prompt files expose the same ordered model set;
- SGLang and vLLM refined-plan templates use the same evidence/checkpoint
  fields where framework-neutral;
- dossier cards require immutable source head, PR state, motivation,
  implementation, validation, and limitations;
- new model config index entries have all dimensions needed by
  `build_layer_ops`.

- [ ] **Step 2: Add only defensible compute templates**

Add MiniMax M3 and Qwen3.6 aliases/configs only after copying dimensions from
their public `config.json` and checking generated operator shapes. Do not infer
Kimi K3 dimensions from the architecture image. Keep Inkling out unless its
complete public config is available and the simulator can represent its
quantization without a false FLOP claim.

- [ ] **Step 3: Refresh SOTA and dossier workflows**

Update source-head language, required manual diff review, open-versus-merged
classification, current framework commands, cleanup expectations, and
validation-evidence fields. Keep the SGLang loop's TokenSpeed competitor and
the vLLM loop's current scoped competitor set unless a verified executable
TokenSpeed path is added to the vLLM workflow.

- [ ] **Step 4: Synchronize prompt copies**

Replace MiniMax M2.7 operational entries with MiniMax M3 only where the
documented GPU topology is authoritative. Add Qwen3.6 only with a verified
hardware shape. Keep the ordinary and Codex-goal variants byte-consistent in
their model tables and commands apart from their intentional orchestration
instructions.

- [ ] **Step 5: Run tests and commit**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider \
  tests/test_model_profiles.py \
  tests/test_model_pr_dossier_quality.py \
  tests/test_sota_humanize_loop_docs.py \
  tests/test_sglang_sota_humanize_loop_docs.py -q
git add skills/model-compute-simulation/references/model-config-index.json \
  skills/model-optimization \
  skills/sglang-sota-humanize-loop \
  skills/vllm-sota-humanize-loop \
  prompts tests
git commit -m "Synchronize model and SOTA workflows"
```

## Task 13: Refresh README, Plugin Metadata, Dependencies, and CI

**Files:**

- Modify: `README.md`
- Modify: `update_prompt.md`
- Modify: `.claude-plugin/marketplace.json`
- Modify: `.claude-plugin/plugin.json` only if validation requires a schema correction
- Modify: `.pre-commit-config.yaml`
- Modify: `.github/workflows/lint.yml`
- Modify: `.github/linters/lychee.toml` only if a verified new public URL needs a narrow rule

- [ ] **Step 1: Update current launch guidance**

Use:

```bash
claude --model opus --permission-mode auto
codex --sandbox danger-full-access --ask-for-approval never
```

Explain that `opus` tracks the current Opus model and that Claude
`bypassPermissions` is intended only for isolated sandboxes. Remove the
redundant undocumented Codex `--yolo` combination. Keep the eleven core skills
badge because the model-history lookup skill is packaged separately from the
eleven table entries.

- [ ] **Step 2: Add top-level marketplace metadata**

Add a top-level `description` matching the plugin's current scope. Do not add
or change a repository license in this refresh.

- [ ] **Step 3: Update verified dependency and action releases**

Set:

```text
actions/checkout@v7
actions/setup-python@v7
isort 8.0.1
ruff-pre-commit v0.16.0
black 26.5.1
codespell 2.4.3
mirrors-clang-format v22.1.8
lychee lychee-v0.24.2
lycheeverse/lychee-action v2.9.0 or its immutable release SHA
```

Keep `pre-commit-hooks v6.0.0` and
`DoozyX/clang-format-lint-action@v0.20`. Set the workflow clang version to 22
so CI and pre-commit agree.

- [ ] **Step 4: Update the refresh checklist**

Replace PR-72-specific wording with the current branch/PR workflow while
retaining all mandatory five-model, MiniMax M3, live-profiler, NCU, cleanup,
remote-test, and CI gates.

- [ ] **Step 5: Run validators and pre-commit**

```bash
claude plugin validate .claude-plugin/plugin.json
claude plugin validate .claude-plugin/marketplace.json
SKIP=no-commit-to-branch pre-commit run --all-files --show-diff-on-failure
```

Expected: both plugin validators pass without the missing marketplace
description warning and pre-commit is clean.

- [ ] **Step 6: Commit**

```bash
git add README.md update_prompt.md .claude-plugin \
  .pre-commit-config.yaml .github
git commit -m "Refresh repository tooling and guidance"
```

## Task 14: Run Complete Local and Artifact Validation

**Files:**

- Modify only files proven incorrect by validation.

- [ ] **Step 1: Run both full Python test entry points**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider -q
```

Expected: all tests pass; the count is greater than the 118-test baseline.

- [ ] **Step 2: Compile Python and check shell syntax**

```bash
python3 -m compileall -q tools tests skills
find skills -type f -name '*.sh' -print0 |
  xargs -0 -n1 bash -n
```

Expected: no syntax errors.

- [ ] **Step 3: Run deterministic query and output smokes**

Run:

```bash
python3 model-pr-optimization-history/scripts/query.py --model qwen36 --framework sglang
python3 skills/model-architecture-diagram/scripts/model_architecture_diagram.py kimi-k3
python3 skills/model-compute-simulation/scripts/model_compute_simulator.py --list-models
  --query cuda --limit 3
  --query server_args --top 3
```

Expected: each exits zero and returns current indexed evidence.

- [ ] **Step 4: Audit dates, heads, paths, PR states, and temporary files**

Run targeted scans for June 27 current-state claims, all superseded SHAs,
`python/sglang/jit_kernel`, retired vLLM flags, caches, generated scratch data,
and private keys. Inspect every match and retain only explicitly historical
provenance.

- [ ] **Step 5: Run final diff checks**

```bash
git diff --check
git status --short
git diff --stat main...HEAD
git diff --name-status main...HEAD
```

Review the complete diff, not only the final commit.

## Task 15: Execute the Required B200, MiniMax M3, Profiler, and NCU Validation

**Files:**

- Modify: `README.md`, relevant `SKILL.md`, or `update_prompt.md` only when runtime evidence disproves existing guidance.
- Do not commit remote artifact directories or model caches.

- [ ] **Step 1: Enter the prescribed B200 environment and record versions**

Use the repository's B200 machine skill, then:

```bash
ssh -i ~/.ssh/id_ed25519 bbuf@216.114.73.196
sudo docker exec -it sglang_bbuf bash
```

Create an artifact root under `/data/bbuf`, record Python, torch, CUDA, SGLang,
GPU, and CLI help evidence.

- [ ] **Step 2: Run five isolated SGLang serving smokes**

Run one at a time:

```text
Qwen/Qwen2.5-0.5B-Instruct
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen2.5-3B-Instruct
Qwen/Qwen2.5-7B-Instruct
Qwen/Qwen3-8B
```

For each model, verify server startup, `/v1/models`, one OpenAI-compatible
request, the benchmark path, server termination, and zero residual GPU memory
before continuing.

- [ ] **Step 3: Validate MiniMax M3 only in its isolated container**

Start `sglang_m3`, run current loader/cache/JIT unit and smoke coverage, and run
full serving only if the official 2×4 B200 disaggregated recipe prerequisites
are already satisfied. Stop the container if this run started it.

- [ ] **Step 4: Capture and analyze a live SGLang profile**

Use a small model to capture separate prefill and decode traces. Run the
repository analyzer and verify nonempty kernel, overlap, and fuse tables. Save
commands and trace paths in the remote artifact root.

- [ ] **Step 5: Run a minimal Nsight Compute smoke**

Compile or use a tiny CUDA matmul and run:

```bash
ncu --set basic --export "$ART/ncu/matmul-basic.ncu-rep" ./matmul_smoke
```

Expected: the report opens with `ncu --import` and contains a captured kernel.

- [ ] **Step 6: Sync the branch snapshot and run remote repository checks**

```bash
python3 -m pytest -q \
  tests/test_model_pr_dossier_quality.py \
  tests/test_open_pr_watch.py \
  tests/test_llm_serving_cookbook_configs.py

python3 skills/llm-serving-auto-benchmark/scripts/validate_cookbook_configs.py \
  --help-dir "$ART/help" \
  skills/llm-serving-auto-benchmark/configs/cookbook-llm/*.yaml
```

- [ ] **Step 7: Clean the machine**

Kill every server started by the validation, remove only the temporary synced
repository, leave evidence artifacts in the named artifact root, stop
`sglang_m3` when appropriate, and verify all eight B200 GPUs report zero MiB
used.

## Task 16: Final Completion Audit and Draft PR

**Files:**

- Modify only files required by a final failed gate.

- [ ] **Step 1: Re-run local release gates after remote-driven corrections**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider -q
claude plugin validate .claude-plugin/plugin.json
claude plugin validate .claude-plugin/marketplace.json
SKIP=no-commit-to-branch pre-commit run --all-files --show-diff-on-failure
git diff --check
```

- [ ] **Step 2: Audit the design acceptance checklist item by item**

For each item in
`docs/superpowers/specs/2026-07-27-comprehensive-refresh-design.md`, link it to
test output, source SHA, generated artifact, remote log, GPU cleanup evidence,
or a current file diff. Fix any item without direct evidence.

- [ ] **Step 3: Commit validation-driven corrections**

```bash
git status --short
git diff --name-only --diff-filter=ACMRTUXB -z | xargs -0 git add --
git diff --cached --check
git commit -m "Address comprehensive refresh validation"
```

Skip this commit when validation required no tracked correction.

- [ ] **Step 4: Verify GitHub prerequisites and push**

```bash
gh --version
gh auth status
git status --short --branch
git push -u origin agent/refresh-ai-infra-skills-july-2026
```

- [ ] **Step 5: Open a draft pull request**

Target `BBuf/AI-Infra-Auto-Driven-SKILLS:main`. The PR body must state:

- all correctness fixes and root causes;
- exact final upstream heads;
- new/removed model and flag coverage;
- generated and manually reviewed evidence;
- local test, plugin, pre-commit, remote B200, MiniMax M3, live-profiler, NCU,
  and cleanup results;
- any environment gap without turning it into a support claim.

- [ ] **Step 6: Watch GitHub checks and repair in-scope failures**

Use `gh pr checks --watch` and inspect failing Actions logs. Apply focused
fixes, rerun the corresponding local gate, commit, push, and wait again until
all required checks are green or a genuinely external pending check is
precisely documented.
