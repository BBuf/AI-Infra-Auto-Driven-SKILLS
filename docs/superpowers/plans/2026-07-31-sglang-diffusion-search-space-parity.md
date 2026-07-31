# SGLang Diffusion Search-Space Parity Implementation Plan

> **For agentic workers:** Execute this plan inline in the current root-agent
> conversation. Do not spawn subagents, nested Codex/Claude sessions, or any
> other AI subprocess.

**Goal:** Give every serial SGLang Diffusion work order a complete,
provenance-bound Sol search-space catalog and stronger SGLang/KDA kernel
knowledge without weakening SGLang-native end-to-end verification.

**Architecture:** Lock primary repositories, derive KDA skill repositories from
the exact KDA gitlinks, snapshot allowlisted text, and build a normalized Sol
catalog during deterministic setup. Route six method families and bind the
catalog plus knowledge manifest into every root-agent work order. Require every
candidate to cite verifiable knowledge provenance.

**Tech Stack:** Python 3.11+, Pydantic v2, TOML, JSON, Git worktrees, pytest,
Ruff.

---

## Task 1: Expand and harden knowledge sources

**Files:**

- Modify: `sgl-engine-sglang-diffusion/knowledge/registry.toml`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/knowledge.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_knowledge.py`

- [ ] **Step 1: Write failing registry coverage tests**

Assert that the registry exposes:

```python
assert set(registry) == {
    "sglang",
    "sol_engine",
    "fastvideo",
    "kda_pilot",
    "kernel_wiki",
    "ncu_report_skill",
    "warp_specialization_report_skill",
}
assert ".claude/skills/add-jit-kernel/**" in registry["sglang"]
assert "search_space/**" in registry["sol_engine"]
assert "candidates/**" in registry["sol_engine"]
assert "diffusion/kernels/**" in registry["kda_pilot"]
assert "sources/**" in registry["kernel_wiki"]
```

Add a test proving that a required source with no matched files raises
`KnowledgeSyncError`.

- [ ] **Step 2: Run the failing tests**

```bash
cd sgl-engine-sglang-diffusion
pytest -q tests/test_knowledge.py
```

Expected: FAIL because Sol and the standalone KDA skill sources are absent and
empty snapshots are currently accepted.

- [ ] **Step 3: Add the allowlists and nonempty invariant**

Add all seven sources to `knowledge/registry.toml`. Expand SGLang to include
root kernel/profiler skills and relevant multimodal runtime surfaces. Expand
KDA-Pilot to `diffusion/**`. Give the three submodule sources independent
allowlists.

In `sync_source`, reject an empty `entries` list:

```python
if not entries:
    raise KnowledgeSyncError(
        f"knowledge source {name!r} matched no allowlisted text files"
    )
```

- [ ] **Step 4: Run focused tests**

```bash
pytest -q tests/test_knowledge.py
```

Expected: PASS.

## Task 2: Derive KDA skill source locks from exact gitlinks

**Files:**

- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/runtime.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_runtime_e2e.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_sources.py`

- [ ] **Step 1: Write failing submodule-lock tests**

Create three tiny Git repositories, add them as these KDA paths, and commit:

```text
external/KernelWiki
external/ncu-report-skill
external/warp-specialization-report-skill
```

Assert that source locking records independent full SHAs named
`kernel_wiki`, `ncu_report_skill`, and
`warp_specialization_report_skill`, and that each SHA equals its parent
gitlink.

- [ ] **Step 2: Run the failing tests**

```bash
pytest -q tests/test_sources.py tests/test_runtime_e2e.py
```

Expected: FAIL because only four primary sources are locked.

- [ ] **Step 3: Implement derived locks**

Add a fixed path-to-source-name map. Read `.gitmodules` and `git ls-tree` from
the locked KDA bare cache, normalize:

```text
git@github.com:owner/repository.git
```

to:

```text
https://github.com/owner/repository.git
```

Lock each derived repository at the exact gitlink SHA. Require all seven
sources on load and verify the derived repository URL and requested SHA against
the locked KDA revision.

- [ ] **Step 4: Run focused tests**

```bash
pytest -q tests/test_sources.py tests/test_runtime_e2e.py
```

Expected: PASS.

## Task 3: Build a normalized Sol search-space catalog

**Files:**

- Create: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/search_space.py`
- Create: `sgl-engine-sglang-diffusion/tests/test_search_space.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/runtime.py`
- Modify: `sgl-engine-sglang-diffusion/contracts/sol_engine/source-lock.json`
- Modify: `sgl-engine-sglang-diffusion/contracts/sol_engine/source-hashes.json`

- [ ] **Step 1: Write catalog tests**

Construct a minimal fake Sol tree containing all six canonical documents, one
structured candidate, one recipe, one registered technique, and one registered
transform. Assert:

```python
assert set(catalog["families"]) == {
    "kernel",
    "cache",
    "sparse_attention",
    "quantization",
    "token_pruning",
    "topology",
}
assert catalog["candidate_count"] == 1
assert catalog["recipe_count"] == 1
assert catalog["families"]["sparse_attention"]["candidates"][0][
    "required_capabilities"
] == ["has_attention_backend_switch"]
```

Also test failure on a missing canonical document, unknown candidate
dimension, and missing `generic_impl`.

- [ ] **Step 2: Run the failing tests**

```bash
pytest -q tests/test_search_space.py
```

Expected: FAIL because the catalog builder does not exist.

- [ ] **Step 3: Implement the catalog builder**

Parse method-family headings and bullets from the six documents. Parse
structured manifests under `candidates/**/*.toml`, top-level recipe manifests,
site documentation paths, and `@register_technique` /
`@register_transform` declarations. Preserve exact source paths, required
capabilities, and source hashes.

Write `SEARCH-SPACE.json` atomically with the locked Sol commit.

- [ ] **Step 4: Bind catalog generation into setup**

After knowledge synchronization, call:

```python
build_sol_search_space_catalog(
    sol_checkout=worktrees["sol_engine"],
    sol_commit=locks["sol_engine"].commit,
    output_path=self.campaign_dir / "SEARCH-SPACE.json",
)
```

Expand the reviewed Sol source contract to cover the canonical search-space
documents and composition/candidate-schema files, then regenerate the checked
hashes from commit
`cee25847afdd34bc656abcca126262200b088dc8`.

- [ ] **Step 5: Run focused tests**

```bash
pytest -q tests/test_search_space.py tests/test_runtime_e2e.py \
  tests/test_techniques.py
```

Expected: PASS.

## Task 4: Route complete families and bind knowledge into work orders

**Files:**

- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/profiler.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/models.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/work_orders.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/integrator.py`
- Modify: `sgl-engine-sglang-diffusion/techniques/registry.toml`
- Move: `sgl-engine-sglang-diffusion/techniques/pisa.md`
- Test: `sgl-engine-sglang-diffusion/tests/test_profiler.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_work_orders.py`
- Test: `sgl-engine-sglang-diffusion/tests/test_techniques.py`

- [ ] **Step 1: Write failing routing and binding tests**

Require the quality-gated route order:

```python
[
    "kernel",
    "cache",
    "sparse_attention",
    "quantization",
    "token_pruning",
]
```

Require `topology` after `kernel` for multi-GPU workloads. Assert that a claimed
work order has valid paths and hashes for `KNOWLEDGE.json` and
`SEARCH-SPACE.json`.

- [ ] **Step 2: Run the failing tests**

```bash
pytest -q tests/test_profiler.py tests/test_work_orders.py \
  tests/test_techniques.py tests/test_integration_flow.py
```

Expected: FAIL on the PISA-only route and missing work-order fields.

- [ ] **Step 3: Replace the PISA lane**

Rename the route, scope, registry entry, and integration rank from `pisa` to
`sparse_attention`. Keep PISA as a named candidate family inside the new scope.

- [ ] **Step 4: Add work-order bindings**

Extend `AgentWorkOrder` with:

```python
knowledge_manifest_path: Path
search_space_path: Path
knowledge_manifest_sha256: str
search_space_sha256: str
```

Require both files before `claim`, compute their hashes, and serialize them
into `AGENT-WORK.json`.

- [ ] **Step 5: Run focused tests**

```bash
pytest -q tests/test_profiler.py tests/test_work_orders.py \
  tests/test_techniques.py tests/test_integration_flow.py
```

Expected: PASS.

## Task 5: Require verifiable knowledge provenance

**Files:**

- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/models.py`
- Modify: `sgl-engine-sglang-diffusion/src/sgl_engine_sglang_diffusion/verifier.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_verifier.py`
- Modify: `sgl-engine-sglang-diffusion/tests/test_review.py`
- Regenerate: `sgl-engine-sglang-diffusion/schemas/candidate.schema.json`

- [ ] **Step 1: Write failing provenance tests**

Use:

```json
{
  "source": "sglang",
  "commit": "40-hex",
  "path": "python/sglang/example.py",
  "sha256": "64-hex"
}
```

Assert that an empty origin list, unknown source, wrong commit, unknown path,
and wrong SHA are rejected. Assert that an exact index entry passes.

- [ ] **Step 2: Run the failing tests**

```bash
pytest -q tests/test_verifier.py tests/test_review.py
```

Expected: FAIL because knowledge origins are currently untyped and unchecked.

- [ ] **Step 3: Implement the provenance model and verifier**

Add:

```python
class KnowledgeOrigin(StrictModel):
    source: str = Field(min_length=1)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
```

Set `CandidateManifest.knowledge_origin` to
`list[KnowledgeOrigin] = Field(min_length=1)`. Resolve `KNOWLEDGE.json`, load
the referenced snapshot indices, and match all four fields.

- [ ] **Step 4: Regenerate schemas and run focused tests**

```bash
python -c 'from pathlib import Path; from sgl_engine_sglang_diffusion.artifacts import write_schemas; write_schemas(Path("schemas"))'
pytest -q tests/test_artifacts.py tests/test_verifier.py tests/test_review.py
```

Expected: PASS.

## Task 6: Expand scopes and the root-agent skill

**Files:**

- Modify: `sgl-engine-sglang-diffusion/techniques/cache.md`
- Modify: `sgl-engine-sglang-diffusion/techniques/kernel.md`
- Create: `sgl-engine-sglang-diffusion/techniques/sparse_attention.md`
- Delete: `sgl-engine-sglang-diffusion/techniques/pisa.md`
- Modify: `sgl-engine-sglang-diffusion/techniques/quantization.md`
- Modify: `sgl-engine-sglang-diffusion/techniques/token_pruning.md`
- Modify: `sgl-engine-sglang-diffusion/techniques/topology.md`
- Modify: `skills/sglang-diffusion-auto-optimize/SKILL.md`
- Modify: `skills/sglang-diffusion-auto-optimize/references/work-order-protocol.md`
- Modify: `skills/sglang-diffusion-auto-optimize/references/request-template.yaml`
- Modify: `sgl-engine-sglang-diffusion/README.md`

- [ ] **Step 1: Expand Cache and sparse-attention scopes**

Remove the closed three-cache-family constraint. Require inspection of the full
Sol family list and at least five applicable cache directions before a
`no_gain` conclusion.

Document all nine Sol sparse-attention families and Sol-Attn under the
quality-gated `sparse_attention` scope.

- [ ] **Step 2: Add the knowledge-first root-agent protocol**

Require the root agent to read the catalog projection, query SGLang and KDA
knowledge snapshots, distinguish documented/referenced/adapted/validated
status, and cite exact knowledge origins in every candidate.

State that kernel skills are loaded only after profile evidence selects a
kernel hypothesis, while their locked indices remain discoverable throughout
the campaign.

- [ ] **Step 3: Validate the updated skill**

Run the repository's skill validator when present and verify:

```bash
rg -n "SEARCH-SPACE.json|KNOWLEDGE.json|sparse_attention|knowledge_origin" \
  skills/sglang-diffusion-auto-optimize sgl-engine-sglang-diffusion
```

Expected: all four concepts appear in the execution contract.

## Task 7: Full validation and publication

**Files:**

- All files modified above.

- [ ] **Step 1: Run format and complete tests**

```bash
cd sgl-engine-sglang-diffusion
python -m pytest -q
python -m compileall -q src tools
git diff --check
```

Expected: all tests pass, compileall returns zero, and `git diff --check`
prints nothing.

- [ ] **Step 2: Inspect the final scope**

```bash
git status -sb
git diff --stat HEAD~1
git diff --check
```

Confirm that only the existing PR's SGLang Diffusion flow, skill, tests,
contracts, and design/plan documents changed.

- [ ] **Step 3: Commit and push**

```bash
git add <explicit-reviewed-paths>
git commit -m "feat: mirror Sol optimization knowledge"
git push -u origin agent/single-agent-diffusion-flow
```

Update the existing draft PR description with the full search-space and
knowledge-provenance changes plus the validation result.
