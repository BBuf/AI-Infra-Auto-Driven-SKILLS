# Diffusion Per-Request Mean E2E Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make per-request mean E2E latency the Controller's sole authoritative optimization metric while preserving the five-request wall time as audited evidence.

**Architecture:** Normalize every raw benchmark into an explicit timing triple: mean E2E, complete workload duration, and successful request count. Version baseline, performance, delivery, and progress artifacts around those names; independently recompute and cross-check the mean in the Driver and Verifier before speedup or target decisions.

**Tech Stack:** Python 3.11, Pydantic 2, pytest, JSON Schema, Markdown contracts, GitHub Actions.

---

### Task 1: Normalize raw benchmark timing into explicit mean and total fields

**Files:**
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/driver.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_driver.py`

- [ ] **Step 1: Write failing normalization tests**

Add a five-request raw result with `total_duration_seconds=400.0` and
`latency_per_request_seconds=80.0`, then require:

```python
assert normalized["mean_e2e_s"] == 80.0
assert normalized["workload_total_s"] == 400.0
assert normalized["request_count"] == 5
assert "total_s" not in normalized
```

Add rejection tests for a reported mean inconsistent with total/count and for
missing successful-request count.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```bash
cd sgl-engine-sglang-diffusion
PYTHONPATH=src pytest -q tests/test_driver.py
```

Expected: assertions fail because the Driver still emits `total_s`.

- [ ] **Step 3: Implement independent mean computation**

Change `normalize_output` to require the exact successful count, extract the
complete duration, compute `mean_e2e_s = workload_total_s / successful`, and
validate any raw `latency_per_request_seconds` with `math.isclose`.

- [ ] **Step 4: Re-run the focused tests**

Run the command from Step 2. Expected: all driver tests pass.

### Task 2: Version baseline, delivery, and integrated performance artifacts

**Files:**
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/models.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/baseline.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/integrator.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/artifacts.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_artifacts.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_integration_flow.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_final_quality.py`
- Modify generated files under: `sgl-engine-sglang-diffusion/schemas/`

- [ ] **Step 1: Add failing model and integration expectations**

Require `BaselineRecord` schema 2 to expose:

```python
mean_e2e_s=80.0
workload_total_s=400.0
request_count=5
```

Require `PerformanceRecord` to expose baseline/candidate mean, totals, and one
frozen request count, and reject the old ambiguous timing field names.

- [ ] **Step 2: Run artifact and integration tests**

```bash
cd sgl-engine-sglang-diffusion
PYTHONPATH=src pytest -q tests/test_artifacts.py tests/test_integration_flow.py tests/test_final_quality.py
```

Expected: failures on missing explicit timing fields.

- [ ] **Step 3: Implement the schema-2 models and producers**

Make baseline and integration consume the Driver's explicit timing triple.
Compute speedup only from baseline/candidate mean values, while retaining both
workload totals in the delivery record.

- [ ] **Step 4: Regenerate schemas and rerun tests**

```bash
cd sgl-engine-sglang-diffusion
PYTHONPATH=src python3 -c 'from pathlib import Path; from sgl_engine_sglang_diffusion.artifacts import write_schemas; write_schemas(Path("schemas"))'
PYTHONPATH=src pytest -q tests/test_artifacts.py tests/test_integration_flow.py tests/test_final_quality.py
```

Expected: generated schemas match and focused tests pass.

### Task 3: Make independent verification recompute the mean

**Files:**
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/verifier.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_verifier.py`

- [ ] **Step 1: Add tamper tests**

Create independently failing cases for candidate mean, complete workload total,
and request-count drift. Each must produce a deterministic verification finding.

- [ ] **Step 2: Run verifier tests and confirm failure**

```bash
cd sgl-engine-sglang-diffusion
PYTHONPATH=src pytest -q tests/test_verifier.py
```

- [ ] **Step 3: Recompute raw benchmark timing in the Verifier**

Require five successful requests, recompute raw mean from total/count, compare
the raw triple with `PERFORMANCE.json` and the delivery record, and calculate
authoritative speedup from mean values only.

- [ ] **Step 4: Re-run verifier tests**

Expected: all valid points pass and every timing tamper fails closed.

### Task 4: Update Controller decisions, progress, prompts, and E2E fixtures

**Files:**
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/controller.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/progress.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/runtime.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_controller.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_progress.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_runtime_e2e.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_runtime_scheduler.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_orchestration.py`

- [ ] **Step 1: Add failing target and rendering expectations**

Require target latency to use `baseline.mean_e2e_s`, and progress output to
render both:

```text
latency     80.0000s/request baseline -> 40.0000s/request integrated
workload    400.0000s/5 requests -> 200.0000s/5 requests
```

- [ ] **Step 2: Run focused controller/runtime tests**

```bash
cd sgl-engine-sglang-diffusion
PYTHONPATH=src pytest -q tests/test_controller.py tests/test_progress.py tests/test_runtime_e2e.py tests/test_runtime_scheduler.py tests/test_orchestration.py
```

- [ ] **Step 3: Update all scientific consumers and fixtures**

Replace ambiguous timing keys in target checks, candidate registry summaries,
executor prompts, progress JSON, and fake Agent deliveries. Preserve the full
five-prompt gate and all non-timing contracts.

- [ ] **Step 4: Re-run the focused suite**

Expected: Controller and E2E tests pass using mean latency throughout.

### Task 5: Update public documentation, skill contract, and standalone prompt

**Files:**
- Modify: `sgl-engine-sglang-diffusion/README.md`
- Modify: `sgl-engine-sglang-diffusion/prompts/executor.md`
- Modify: `sgl-engine-sglang-diffusion/prompts/master.md`
- Modify: `skills/sglang-diffusion-auto-optimize/SKILL.md`
- Modify: `skills/sglang-diffusion-auto-optimize/references/progress-contract.md`
- Modify after merge: `/Users/bbuf/工作目录/Common/prompt.md`

- [ ] **Step 1: Replace ambiguous total-latency language**

Document that mean E2E is authoritative, workload total is audit-only, exactly
five prompts remain mandatory, and speedups are never accumulated.

- [ ] **Step 2: Search for stale field names and semantics**

```bash
rg -n "baseline_total_s|candidate_total_s|integrated_total_s|frozen_baseline_total_s|total_s" \
  sgl-engine-sglang-diffusion skills/sglang-diffusion-auto-optimize
```

Expected: only explicitly documented obsolete-schema rejection or unrelated
wall-clock concepts remain.

### Task 6: Validate, publish, merge, and install

**Files:**
- Modify test fixtures only if the complete suite exposes legitimate stale
  timing fields.

- [ ] **Step 1: Run complete local validation**

```bash
cd sgl-engine-sglang-diffusion
PYTHONPATH=src pytest -q
PYTHONPATH=src python3 -m compileall -q src tests
ruff check src tests
```

From the repository root, run changed-file pre-commit checks and
`git diff --check`, then build a wheel into a temporary directory.

- [ ] **Step 2: Commit and publish**

Commit the implementation, push `agent/mean-e2e-timing`, open a draft PR with
the timing-contract rationale and validation results, wait for required CI, mark
ready, and squash merge.

- [ ] **Step 3: Reinstall and verify the local skill**

Back up the current installed skill, install from the exact merge commit, and
require `diff -qr` against the merged repository source to succeed.

- [ ] **Step 4: Update and validate `prompt.md`**

Use the merged commit and tool paths, replace five-prompt-total target wording
with per-request-mean wording, parse the embedded launch request, and confirm
the 5.0 target and frozen command remain unchanged.
