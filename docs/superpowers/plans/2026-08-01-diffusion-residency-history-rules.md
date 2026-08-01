# Diffusion Residency and Historical Rules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use writing-plans discipline to execute this plan task-by-task with a failing test before each behavior change.

**Goal:** Make the SGLang Diffusion optimization flow search measured high-VRAM residency choices and reusable, diff-reviewed historical optimization rules, compose their wins, and require profile-bound evidence before acceptance.

**Architecture:** A validated TOML history catalog supplies lane-owned hypotheses to executor prompts. A new lossless `residency` lane owns placement/offload search and emits a typed evidence artifact checked by the independent verifier. Existing kernel search expands to compiler, VAE/output, and scheduler/runtime E2E hotspots while quality-changing rules remain in their gated lanes.

**Tech Stack:** Python 3.11, Pydantic 2, TOML (`tomllib`), pytest, Markdown contracts, JSON Schema, Git/GitHub CLI.

---

## Task 1: Add the historical-rule catalog contract

**Files:**

- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/history_rules.py`
- Create: `sgl-engine-sglang-diffusion/knowledge/history-rules.toml`
- Create: `sgl-engine-sglang-diffusion/tests/test_history_rules.py`

**Steps:**

1. Write tests that load a complete rule catalog and reject duplicate IDs, unknown technique owners, malformed PR URLs or merge commits, empty trigger/action/evidence lists, and correctness drift from `techniques/registry.toml`.
2. Run `pytest -q tests/test_history_rules.py` from `sgl-engine-sglang-diffusion` and confirm the import or assertions fail.
3. Implement immutable source/rule records and `HistoryRuleCatalog.load`, `for_technique`, and deterministic lane rendering with a SHA-256 catalog digest.
4. Populate representative rules for residency, compile/graph, VAE/output, runtime/synchronization, exact reuse, and kernel fusion. Every source must point to a manually reviewed merged SGLang PR and full merge commit.
5. Re-run the focused test and `python -m compileall -q src`.

## Task 2: Add and route the residency lane

**Files:**

- Create: `sgl-engine-sglang-diffusion/techniques/residency.md`
- Modify: `sgl-engine-sglang-diffusion/techniques/registry.toml`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/profiler.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/integrator.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_techniques.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_profiler.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_integration_flow.py`

**Steps:**

1. Add failing expectations that `residency` is the first default lossless lane, has the five reviewed coverage IDs, is routed on profiled workloads, and precedes kernel in canonical integration order.
2. Run the three focused test modules and confirm failures.
3. Define the lane contract for component residency, partial DiT residency, prefetch, compile-time placement, and load-order lifetime. Require measured headroom and frozen topology; prohibit a copied VRAM threshold as acceptance evidence.
4. Route residency before kernel without modifying the baseline's parallel degrees or selected GPU UUIDs.
5. Re-run the focused tests.

## Task 3: Inject only lane-owned historical rules

**Files:**

- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/runtime.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_runtime_e2e.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_orchestration.py`

**Steps:**

1. Add a failing executor-prompt test asserting that the residency prompt contains residency rules and the catalog digest but excludes kernel-owned rules.
2. Load the checked-in catalog once per prompt construction, render only the active technique's subset, and insert it as provenance-addressed auxiliary knowledge.
3. Preserve existing prompt precedence and locked external knowledge sections.
4. Re-run the focused tests.

## Task 4: Add typed residency evidence and fail-closed verification

**Files:**

- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/models.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/artifacts.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/verifier.py`
- Create: `sgl-engine-sglang-diffusion/schemas/residency-evidence.schema.json`
- Modify: `sgl-engine-sglang-diffusion/tests/helpers.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_artifacts.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_verifier.py`

**Steps:**

1. Write verifier tests for valid profile-bound evidence and failures for missing evidence, candidate/run mismatch, stale profile hash, GPU-set drift, negative safety headroom, contradictory component/layer strategy, placement not restored, zero engagement, and absent full-run/equivalence artifacts.
2. Add strict Pydantic models for memory snapshots, transfer measurements, component placement, conflict checks, and the top-level residency artifact.
3. Require `RESIDENCY-EVIDENCE.json` for accepted residency frontier points and verify every referenced artifact hash/path inside the candidate run directory.
4. Generate the new public schema through `write_schemas` and update exact schema-set tests.
5. Re-run artifact and verifier tests.

## Task 5: Expand lossless E2E kernel and knowledge coverage

**Files:**

- Modify: `sgl-engine-sglang-diffusion/techniques/kernel.md`
- Modify: `sgl-engine-sglang-diffusion/techniques/registry.toml`
- Modify: `sgl-engine-sglang-diffusion/knowledge/registry.toml`
- Modify: `sgl-engine-sglang-diffusion/tests/test_techniques.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_knowledge.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_runtime_scheduler.py`

**Steps:**

1. Add failing checks for `compile-graph-warmup`, `vae-decode-postprocess`, and `scheduler-precompute-sync`, plus source paths for memory managers, pipeline stages, VAE, schedulers, distributed runtime, output utilities, and torch compilation.
2. Expand the kernel contract from repeated DiT-only work to the full load-excluded E2E scope, while explicitly keeping precision/approximation and topology changes in their existing gates.
3. Increase the kernel round budget proportionally and ensure a disposition must cover every old and new family rather than closing after one failed hypothesis.
4. Re-run the focused tests.

## Task 6: Check in the manually reviewed PR dossier

**Files:**

- Create: `docs/references/sglang-diffusion-pr-rule-audit.md`

**Steps:**

1. For every PR cited in `history-rules.toml`, fetch PR metadata and inspect the full merge diff from the local SGLang object database or GitHub.
2. Record title, state, merge time, additions/deletions/file count, reviewed diff size, motivation, concrete files/symbols, a short real diff excerpt, validation result, and safe generalization boundary.
3. State that historical thresholds are hypothesis seeds, not acceptance gates, and distinguish exact/lossless rules from quality-gated rules.
4. Cross-check catalog PR URLs and commits against the dossier with a test or deterministic script.

## Task 7: Update flow documentation and skill behavior

**Files:**

- Modify: `sgl-engine-sglang-diffusion/README.md`
- Modify: `sgl-engine-sglang-diffusion/prompts/executor.md`
- Modify: `sgl-engine-sglang-diffusion/prompts/master.md`
- Modify: `sglang-diffusion-auto-optimize/SKILL.md`
- Modify: `sglang-diffusion-auto-optimize/references/progress-contract.md`
- Modify related skill references discovered by search.

**Steps:**

1. Document residency routing, high-memory preflight, typed evidence, active-lane historical rules, expanded kernel coverage, and composition requirements.
2. Explain that Sol-Engine material may be used through the new controller flow without cloning or invoking the legacy flow.
3. Ensure completion still requires the integrated target or an independent exhausted-search certificate; one rejected technique never closes the campaign.
4. Search for stale statements that limit kernel work to DiT or treat offload thresholds as universal and correct them.

## Task 8: Run complete validation and package checks

**Files:**

- Modify any test fixtures legitimately broken by the new default lane.

**Steps:**

1. Run `pytest -q` from `sgl-engine-sglang-diffusion`.
2. Run `python -m compileall -q src tests`.
3. Run `git diff --check` and search new files for `TODO`, `TBD`, or placeholder claims.
4. Build the wheel with `python -m build` when the build module is available; otherwise run `python -m pip wheel --no-deps .` into a temporary directory.
5. Inspect the final diff and commit implementation/docs in logical commits.

## Task 9: Publish, merge, reinstall, and update the standalone prompt

**Files:**

- Modify outside the repository: `/Users/bbuf/工作目录/Common/prompt.md`
- Replace installed skill: `/Users/bbuf/.codex/skills/sglang-diffusion-auto-optimize`

**Steps:**

1. Push `agent/diffusion-residency-history-rules`, open a draft PR with evidence and test results, monitor CI, mark ready, and squash-merge after required checks pass.
2. Back up the installed skill, install from the merged main commit, and byte-compare it with the repository source; restore the backup on failure.
3. With `apply_patch`, replace every target-semantic `3×`, `3.0`, `3.00`, and `3x` occurrence in `prompt.md` with `5×`, `5.0`, `5.00`, and `5x` respectively.
4. Validate the prompt launch request parses and contains a 5.0 target, and verify no stale 3× completion condition remains.
5. Report the merged PR URL, merge commit, local skill verification, prompt path, and final test/CI status.
