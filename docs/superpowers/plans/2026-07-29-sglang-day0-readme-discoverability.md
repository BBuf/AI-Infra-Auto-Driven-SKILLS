# SGLang Day-0 Skill Discoverability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the merged SGLang model Day-0 support skill visible, installable, and consistently versioned through repository documentation and Claude plugin metadata.

**Architecture:** Preserve the existing README structure and add the new skill at the same entry points used by other core skills. Enforce the public skill counts, install commands, repository-map entry, and matching plugin versions through the existing repository metadata test.

**Tech Stack:** Markdown, JSON, pytest, pre-commit.

---

## File Map

- Modify `README.md`: overview, core count/table, plugin count, installation
  commands, invocation examples, and repository map.
- Modify `skills/model-optimization/README.md`: document both shared
  model-optimization skills.
- Modify `.claude-plugin/plugin.json`: add Day-0 support to the description and
  set version `0.2.0`.
- Modify `.claude-plugin/marketplace.json`: mirror the description and version
  update.
- Modify `tests/test_repository_metadata.py`: enforce documentation and plugin
  metadata consistency.
- Modify `tests/test_model_pr_dossier_quality.py`: update the plugin skill count
  contract shared with the dossier documentation test.

### Task 1: Add failing metadata contracts

**Files:**

- Modify: `tests/test_repository_metadata.py`
- Modify: `tests/test_model_pr_dossier_quality.py`

- [ ] **Step 1: Update the README count assertion**

Replace:

```python
assert "core_skills-11" in readme
```

with:

```python
assert "core_skills-12" in readme
assert "After reload, the 13 skills appear" in readme
```

- [ ] **Step 2: Add discoverability assertions**

Add:

```python
def test_sglang_day0_skill_is_discoverable_and_installable() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    skill_path = "skills/model-optimization/sglang-model-day0-support"

    assert "[`sglang-model-day0-support`]" in readme
    assert (
        f'ln -s "$PWD/{skill_path}" '
        "~/.claude/skills/sglang-model-day0-support"
    ) in readme
    assert (
        f"cp -R {skill_path} "
        "<agent-skill-dir>/sglang-model-day0-support"
    ) in readme
    assert "└── sglang-model-day0-support/" in readme
```

- [ ] **Step 3: Add matching plugin-version assertions**

Extend the marketplace test to load both metadata files:

```python
plugin = json.loads(
    (ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
)
assert marketplace["plugins"][0]["version"] == "0.2.0"
assert plugin["version"] == "0.2.0"
assert marketplace["plugins"][0]["version"] == plugin["version"]
assert "Day-0" in marketplace["description"]
assert "Day-0" in plugin["description"]
```

- [ ] **Step 4: Update the shared plugin-count contract**

In `tests/test_model_pr_dossier_quality.py`, replace:

```python
assert "After reload, the 12 skills appear" in readme
```

with:

```python
assert "After reload, the 13 skills appear" in readme
```

- [ ] **Step 5: Run the focused tests and confirm failure**

Run:

```bash
pytest -q tests/test_repository_metadata.py tests/test_model_pr_dossier_quality.py
```

Expected: failures for the old core/plugin counts, missing install entry, and
old plugin versions.

### Task 2: Update repository documentation

**Files:**

- Modify: `README.md`
- Modify: `skills/model-optimization/README.md`

- [ ] **Step 1: Update the root overview and badge**

Add SGLang model Day-0 support to the centered subtitle and opening capability
paragraph. Change:

```text
core_skills-11
```

to:

```text
core_skills-12
```

- [ ] **Step 2: Add the core-skill table entry**

Insert after `model-pr-diff-dossier`:

```markdown
| [`sglang-model-day0-support`](skills/model-optimization/sglang-model-day0-support/) | You need to turn a new SGLang model architecture into a public Day-0 PR DAG, parallel/kernel adaptation plan, seven-gate validation matrix, release lock, and sanitized evidence bundle. |
```

- [ ] **Step 3: Add the installation and invocation entries**

Add to the symlink block:

```bash
ln -s "$PWD/skills/model-optimization/sglang-model-day0-support" ~/.claude/skills/sglang-model-day0-support
```

Add to the generic copy block:

```bash
cp -R skills/model-optimization/sglang-model-day0-support <agent-skill-dir>/sglang-model-day0-support
```

Add `[$sglang-model-day0-support]` beside `[$model-pr-diff-dossier]` in the
invocation examples.

- [ ] **Step 4: Update the plugin count and repository map**

Change:

```text
After reload, the 12 skills appear
```

to:

```text
After reload, the 13 skills appear
```

Represent the model-optimization subtree as:

```text
└── model-optimization/
    ├── model-pr-diff-dossier/       # shared PR history quality standard
    └── sglang-model-day0-support/   # model Day-0 PR and release gates
```

- [ ] **Step 5: Update the model-optimization index**

Change the title to:

```markdown
# Model Optimization Skills and Standards
```

State that the directory contains shared workflows rather than per-model
skills. Keep the existing dossier bullet and add:

```markdown
- `sglang-model-day0-support/`: evidence-driven workflow for architecture gap
  maps, parallel/kernel adaptation, public PR DAGs, seven release gates, and
  sanitized Day-0 support bundles.
```

### Task 3: Update Claude plugin metadata

**Files:**

- Modify: `.claude-plugin/plugin.json`
- Modify: `.claude-plugin/marketplace.json`

- [ ] **Step 1: Bump both versions**

Set both version fields to:

```json
"version": "0.2.0"
```

- [ ] **Step 2: Update the top-level descriptions**

Use this marketplace description:

```json
"description": "Agent-ready LLM serving, SGLang model Day-0 support, profiling, capacity, SOTA optimization, incident triage, architecture, and PR-history skills."
```

Use this plugin description:

```json
"description": "Agent-ready playbooks for LLM serving benchmarks, SGLang model Day-0 support, capacity planning, torch-profiler triage, pipeline analysis, compute simulation, SGLang/vLLM SOTA Humanize loops, human code review, production incident triage, and model PR-history dossiers."
```

- [ ] **Step 3: Update the marketplace plugin description**

Use:

```json
"description": "LLM serving benchmarks, SGLang model Day-0 support, capacity planning, torch-profiler triage, pipeline analysis, compute simulation, SGLang/vLLM SOTA Humanize loops, code review, prod incident triage, and PR-history dossiers."
```

Preserve all existing tags and add `"day0"` after `"sglang"`.

### Task 4: Validate and publish directly to main

**Files:**

- Modify only the six implementation files listed in the file map.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest -q tests/test_repository_metadata.py tests/test_model_pr_dossier_quality.py
```

Expected: all repository metadata tests pass.

- [ ] **Step 2: Run full repository formatting checks**

Run:

```bash
pre-commit run --all-files
git diff --check
```

Expected: all hooks pass and the diff has no whitespace errors.

- [ ] **Step 3: Review scope**

Run:

```bash
git status --short
git diff --stat
```

Expected: only `README.md`, `skills/model-optimization/README.md`, both plugin
metadata files, `tests/test_repository_metadata.py`, and
`tests/test_model_pr_dossier_quality.py` are modified beyond the already
committed design and plan documents.

- [ ] **Step 4: Commit**

Run:

```bash
git add README.md skills/model-optimization/README.md \
  .claude-plugin/plugin.json .claude-plugin/marketplace.json \
  tests/test_repository_metadata.py tests/test_model_pr_dossier_quality.py \
  docs/superpowers/plans/2026-07-29-sglang-day0-readme-discoverability.md
git commit -m "docs: expose SGLang Day-0 support skill"
```

- [ ] **Step 5: Push main**

Run:

```bash
git push origin main
```

Expected: `origin/main` advances to the documentation commit.
