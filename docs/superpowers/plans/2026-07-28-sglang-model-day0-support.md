# SGLang Model Day-0 Support Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable SGLang model Day-0 support skill with Kimi K3 and DeepSeek V4 public case studies, deterministic evidence/bundle audits, copyable templates, and a worked synthetic demonstration.

**Architecture:** Keep the operational workflow in a concise `SKILL.md`; load the Day-0 contract, evidence policy, sanitization policy, and two model case studies from flat references only when needed. Use standard-library Python scripts for mechanical GitHub evidence collection and fail-closed bundle validation, while preserving manual judgment for all PR motivation and implementation conclusions.

**Tech Stack:** Markdown skills and references, YAML agent metadata, Python 3 standard library, `gh` CLI, pytest, pre-commit.

---

## File Map

- Create `skills/model-optimization/sglang-model-day0-support/SKILL.md`: trigger metadata, staged workflow, reference routing, commands, and completion contract.
- Create `skills/model-optimization/sglang-model-day0-support/agents/openai.yaml`: UI display metadata.
- Create `skills/model-optimization/sglang-model-day0-support/references/day0-contract.md`: capability taxonomy, evidence classes, artifacts, and seven gates.
- Create `skills/model-optimization/sglang-model-day0-support/references/evidence-audit.md`: manual PR-diff card requirements and public evidence syntax.
- Create `skills/model-optimization/sglang-model-day0-support/references/sanitization.md`: clean-room publication rules and forbidden data classes.
- Create `skills/model-optimization/sglang-model-day0-support/references/kimi-k3-case-study.md`: public-only Kimi K3 Day-0 and follow-up lessons.
- Create `skills/model-optimization/sglang-model-day0-support/references/deepseek-v4-case-study.md`: public-only DeepSeek V4 staged-support and repair-loop lessons.
- Create eight files under `skills/model-optimization/sglang-model-day0-support/assets/day0-bundle/`: copyable Day-0 bundle templates.
- Create `skills/model-optimization/sglang-model-day0-support/scripts/collect_public_pr_evidence.py`: public PR metadata/file inventory collector.
- Create `skills/model-optimization/sglang-model-day0-support/scripts/validate_day0_bundle.py`: fail-closed bundle and sanitization validator.
- Create `tests/test_sglang_model_day0_support.py`: collector, validator, and documentation contract tests.
- Create eight resolved demonstration files under `docs/assets/sglang-model-day0-support-demo/`: inspectable fictional-model output.
- Modify no existing model history documents; link to their diff-reviewed public PR cards.

### Task 1: Initialize the skill scaffold

**Files:**
- Create: `skills/model-optimization/sglang-model-day0-support/SKILL.md`
- Create: `skills/model-optimization/sglang-model-day0-support/agents/openai.yaml`
- Create directories: `references/`, `scripts/`, `assets/day0-bundle/`

- [ ] **Step 1: Run the official initializer**

Run:

```bash
python3 "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/init_skill.py" \
  sglang-model-day0-support \
  --path skills/model-optimization \
  --resources scripts,references,assets \
  --interface 'display_name=SGLang Model Day-0 Support' \
  --interface 'short_description=Plan and audit model Day-0 support for SGLang' \
  --interface 'default_prompt=Use $sglang-model-day0-support to design and validate an auditable SGLang Day-0 support bundle for this model.'
```

Expected: the skill directory, `SKILL.md`, and `agents/openai.yaml` are created.

- [ ] **Step 2: Create the nested template directory**

Run:

```bash
mkdir -p skills/model-optimization/sglang-model-day0-support/assets/day0-bundle
```

Expected: the directory exists and no example placeholder files were generated.

- [ ] **Step 3: Verify the scaffold**

Run:

```bash
find skills/model-optimization/sglang-model-day0-support -maxdepth 3 -type f -print | sort
```

Expected: only `SKILL.md` and `agents/openai.yaml` exist before content is added.

### Task 2: Define executable skill behavior and core references

**Files:**
- Modify: `skills/model-optimization/sglang-model-day0-support/SKILL.md`
- Create: `skills/model-optimization/sglang-model-day0-support/references/day0-contract.md`
- Create: `skills/model-optimization/sglang-model-day0-support/references/evidence-audit.md`
- Create: `skills/model-optimization/sglang-model-day0-support/references/sanitization.md`

- [ ] **Step 1: Write the Day-0 contract**

Use these top-level sections in `day0-contract.md`:

```markdown
# SGLang Model Day-0 Contract

## Contents
## Required Bundle
## Capability Taxonomy
## Evidence Classes
## Seven Gates
## PR DAG Rules
## Risk-Pair Selection
## Completion Rules
```

Define the eight required bundle files, the architecture/serving/platform
capability taxonomy, and exactly four evidence classes:
`day0-required`, `post-day0-fix`, `performance-only`, and
`experiment-or-revert`. State that successful startup alone does not close any
gate after the load gate.

- [ ] **Step 2: Write the evidence audit**

Use this exact evidence-record format:

```markdown
- Evidence: https://github.com/sgl-project/sglang/pull/12345 | state: merged | head: 0123456789abcdef0123456789abcdef01234567 | limitation: validated only on the hardware and workload described by the public PR
```

Require each technical PR citation to include or link to a manually reviewed
card containing state, immutable head, diff size, motivation, implementation,
real excerpt, reviewed files, validation, and limitations. Explicitly forbid
scripts from generating motivation or conclusions.

- [ ] **Step 3: Write the sanitization policy**

Use these sections:

```markdown
# Clean-Room Publication Policy

## Publication Boundary
## Allowed Evidence
## Forbidden Data
## Internal-to-Public Mapping
## Performance Claims
## Open, Closed, Reverted, and Experimental Work
## Final Scan
```

Require public upstream URLs and public source paths. Forbid private repository
and PR identifiers, internal branch/commit names, people, machines, IPs,
absolute work paths, private registries, artifact digests, and experiment
round identifiers. Define the mapping as ephemeral and uncommitted.

- [ ] **Step 4: Replace the generated `SKILL.md`**

Use frontmatter containing only:

```yaml
---
name: sglang-model-day0-support
description: Build or audit an evidence-driven SGLang Day-0 support program for a new LLM, VLM, MoE, hybrid-attention, or speculative-decoding model. Use when Codex needs to map a model architecture into SGLang runtime work, design a public support PR DAG, create validation and release gates, sanitize private development evidence, distinguish Day-0 requirements from later fixes or optimizations, or review whether an existing model-support PR is release-ready.
---
```

Write imperative sections for:

```markdown
# SGLang Model Day-0 Support
## Start Here
## Workflow
### 1. Lock the release cut
### 2. Build the architecture gap map
### 3. Classify the work
### 4. Design the PR DAG
### 5. Build the validation matrix
### 6. Execute the seven gates
### 7. Synthesize the public PR
### 8. Track post-Day-0 work
### 9. Validate and sanitize
## Reference Routing
## Commands
## Completion Contract
```

Route users to the Day-0 contract for all tasks, the evidence reference when
using PRs, the sanitization reference whenever non-public inputs exist, and
only the relevant model case study. Keep the file under 500 lines.

- [ ] **Step 5: Validate the initial skill**

Run:

```bash
python3 "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py" \
  skills/model-optimization/sglang-model-day0-support
```

Expected: `Skill is valid!`

- [ ] **Step 6: Commit the core workflow**

Run:

```bash
git add skills/model-optimization/sglang-model-day0-support
git commit -m "feat: add SGLang model Day-0 workflow"
```

### Task 3: Add the two public model case studies

**Files:**
- Create: `skills/model-optimization/sglang-model-day0-support/references/kimi-k3-case-study.md`
- Create: `skills/model-optimization/sglang-model-day0-support/references/deepseek-v4-case-study.md`

- [ ] **Step 1: Write the Kimi K3 case study**

Use these sections:

```markdown
# Kimi K3 Public Day-0 Case Study
## Contents
## Public Evidence Boundary
## Architecture Delta
## Day-0 Support Spine
## Immediate Public Follow-ups
## Hardware and Packaging Extensions
## Failure and Revert Lessons
## Reusable PR Design Lessons
## Detailed Public Dossier
```

Anchor implementation claims to public
`https://github.com/sgl-project/sglang/pull/32541`. Cover the public cookbook,
branch-source Docker fix, remote speculative-draft loading, parser edge cases
and auto-detection, AMD CI/image work, NPU extension, KDA MTP optimization,
TP16/32 shape coverage, portable ROCm sampling fallback, SiTU parameters, and
VLM compatibility. Mark open work as open at the 2026-07-28 audit date.

Link detailed claims to
`../../../../model-pr-optimization-history/sglang/kimi/README.en.md` and
`README.zh.md`. Include no private identifiers.

- [ ] **Step 2: Write the DeepSeek V4 case study**

Use these sections:

```markdown
# DeepSeek V4 Public Day-0 Case Study
## Contents
## Public Evidence Boundary
## Pre-Merge Release Preparation
## Mainline Support
## Immediate Backfill
## Post-Day-0 Repair Categories
## Default Flips and Reverts
## Reusable PR Design Lessons
## Detailed Public Dossier
```

Explain that public cookbook/hardware verification and Docker workflows
preceded the main support merge. Anchor mainline support to
`https://github.com/sgl-project/sglang/pull/23882` and the immediate backfill to
`https://github.com/sgl-project/sglang/pull/24793`. Cover model/config, DSA and
compressed state, SWA, mHC, MTP, parser/tool protocol, quantization,
PP/PD/CP/DP, HiCache, platform ports, graph capture, and later repair loops.

Link the full public inventory to
`../../../../model-pr-optimization-history/sglang/deepseek-v4/README.en.md` and
`README.zh.md` instead of duplicating 171 cards.

- [ ] **Step 3: Audit every cited PR**

For every direct PR link in the two case studies, confirm that either:

1. the case study provides the required evidence record and concrete
   diff-derived detail; or
2. the surrounding sentence links the corresponding diff-reviewed card in the
   existing model history.

Run:

```bash
rg -n 'github.com/sgl-project/sglang/pull/' \
  skills/model-optimization/sglang-model-day0-support/references
```

Expected: every PR URL is public and uses the `sgl-project/sglang` repository.

- [ ] **Step 4: Scan for private identifiers**

Run:

```bash
private_pattern='/[U]sers/|/[h]''ome/|/[d]''ata/|git''@|dkr\\.ecr|\\b(?:[0-9]{1,3}\\.){3}[0-9]{1,3}\\b'
rg -n "$private_pattern" \
  skills/model-optimization/sglang-model-day0-support || true
```

Expected: no output.

- [ ] **Step 5: Commit the case studies**

Run:

```bash
git add skills/model-optimization/sglang-model-day0-support/references
git commit -m "docs: add Kimi K3 and DeepSeek V4 Day-0 cases"
```

### Task 4: Add copyable templates and the worked demonstration

**Files:**
- Create: `skills/model-optimization/sglang-model-day0-support/assets/day0-bundle/scope-contract.md`
- Create: `skills/model-optimization/sglang-model-day0-support/assets/day0-bundle/architecture-gap-map.md`
- Create: `skills/model-optimization/sglang-model-day0-support/assets/day0-bundle/pr-dag.md`
- Create: `skills/model-optimization/sglang-model-day0-support/assets/day0-bundle/validation-matrix.md`
- Create: `skills/model-optimization/sglang-model-day0-support/assets/day0-bundle/release-lock.md`
- Create: `skills/model-optimization/sglang-model-day0-support/assets/day0-bundle/pr-body.md`
- Create: `skills/model-optimization/sglang-model-day0-support/assets/day0-bundle/follow-up-ledger.md`
- Create: `skills/model-optimization/sglang-model-day0-support/assets/day0-bundle/sanitization-report.md`
- Create matching resolved files under: `docs/assets/sglang-model-day0-support-demo/`

- [ ] **Step 1: Write the eight templates**

Start each template with its required heading:

```text
# Day-0 Scope Contract
# Architecture Gap Map
# Pull Request DAG
# Validation Matrix
# Release Lock
# Public Pull Request Body
# Post-Day-0 Follow-up Ledger
# Sanitization Report
```

Use `{{MODEL_ID}}`, `{{SGLANG_SHA}}`, and similarly named braces for values the
user must replace. Include the exact evidence-record syntax from
`evidence-audit.md`.

- [ ] **Step 2: Write the resolved synthetic demonstration**

Create a fictional `Aurora-Hybrid-70B-VL` bundle with no unresolved braces. Use
public precedent records for Kimi K3 and DeepSeek V4, mark Kimi K3 evidence
open with an immutable 40-character public head and a limitation, and mark
DeepSeek V4 evidence merged.

The architecture map must identify these six audit findings:

```text
streaming reasoning/tool marker fragmentation
remote speculative draft loading
recurrent-state transfer in PD
CUDA Graph padding sentinels
multimodal image packaging
post-Day-0 ownership
```

The PR DAG must separate public infrastructure, model spine, protocol/VLM,
platform/packaging, and validation/docs work.

- [ ] **Step 3: Confirm template/demo parity**

Run:

```bash
for name in scope-contract architecture-gap-map pr-dag validation-matrix release-lock pr-body follow-up-ledger sanitization-report; do
  test -f "skills/model-optimization/sglang-model-day0-support/assets/day0-bundle/${name}.md"
  test -f "docs/assets/sglang-model-day0-support-demo/${name}.md"
done
```

Expected: exit status 0.

- [ ] **Step 4: Commit templates and demo**

Run:

```bash
git add skills/model-optimization/sglang-model-day0-support/assets \
  docs/assets/sglang-model-day0-support-demo
git commit -m "docs: add Day-0 bundle templates and demo"
```

### Task 5: Test-drive the public evidence collector

**Files:**
- Create: `tests/test_sglang_model_day0_support.py`
- Create: `skills/model-optimization/sglang-model-day0-support/scripts/collect_public_pr_evidence.py`

- [ ] **Step 1: Write failing collector tests**

Start the test module with:

```python
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = (
    ROOT / "skills" / "model-optimization" / "sglang-model-day0-support"
)


def load_script(name: str):
    path = SKILL_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


collector = load_script("collect_public_pr_evidence")

PR_PAYLOAD = {
    "number": 12345,
    "html_url": "https://github.com/sgl-project/sglang/pull/12345",
    "title": "Add example model",
    "state": "closed",
    "draft": False,
    "created_at": "2026-07-27T00:00:00Z",
    "updated_at": "2026-07-28T00:00:00Z",
    "closed_at": "2026-07-28T00:00:00Z",
    "merged_at": "2026-07-28T00:00:00Z",
    "head": {"sha": "a" * 40},
    "base": {"sha": "b" * 40},
    "additions": 12,
    "deletions": 3,
    "changed_files": 1,
}
FILE_PAYLOAD = [
    {
        "filename": "python/sglang/srt/models/example.py",
        "status": "added",
        "additions": 12,
        "deletions": 3,
        "changes": 15,
    }
]
```

Then add these tests:

```python
def test_parse_public_pr_url_accepts_allowed_repo():
    parsed = collector.parse_pr_url(
        "https://github.com/sgl-project/sglang/pull/23882",
        {"sgl-project/sglang"},
    )
    assert parsed.repository == "sgl-project/sglang"
    assert parsed.number == 23882


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/sgl-project/sglang/pull/23882",
        "https://example.com/sgl-project/sglang/pull/23882",
        "https://github.com/private/example/pull/1",
    ],
)
def test_parse_public_pr_url_rejects_unapproved_inputs(url):
    with pytest.raises(ValueError):
        collector.parse_pr_url(url, {"sgl-project/sglang"})


def test_build_record_is_mechanical_only():
    record = collector.build_record(PR_PAYLOAD, FILE_PAYLOAD, captured_at="2026-07-28T00:00:00Z")
    assert record["state"] == "merged"
    assert record["head_sha"] == "a" * 40
    assert record["files"][0]["filename"] == "python/sglang/srt/models/example.py"
    assert "motivation" not in record
    assert "implementation" not in record
    assert "performance" not in record
```

Load the script through `importlib.util.spec_from_file_location`, following
existing repository test patterns.

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
pytest -q tests/test_sglang_model_day0_support.py -k 'parse_public_pr_url or build_record'
```

Expected: failure because the collector module does not exist.

- [ ] **Step 3: Implement the collector**

Implement:

```python
@dataclasses.dataclass(frozen=True)
class PublicPR:
    repository: str
    number: int
    url: str


def parse_pr_url(url: str, allowed_repositories: set[str]) -> PublicPR:
    match = re.fullmatch(
        r"https://github\\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)/?",
        url,
    )
    if match is None:
        raise ValueError(f"not a canonical public GitHub PR URL: {url}")
    repository = f"{match.group(1)}/{match.group(2)}"
    if repository not in allowed_repositories:
        raise ValueError(f"repository is not allowed: {repository}")
    return PublicPR(repository, int(match.group(3)), url.rstrip("/"))
```

Use `gh api` through `subprocess.run(..., check=True, text=True,
capture_output=True)` for PR metadata and paginated files. Build a JSON object
with schema version, capture time, mechanical metadata, and sorted file
inventory. Do not include PR body, motivation, implementation, or performance
fields. Write output atomically only after all requested PRs succeed.

- [ ] **Step 4: Run collector tests**

Run:

```bash
pytest -q tests/test_sglang_model_day0_support.py -k 'parse_public_pr_url or build_record'
```

Expected: all selected tests pass.

### Task 6: Test-drive the Day-0 bundle validator

**Files:**
- Modify: `tests/test_sglang_model_day0_support.py`
- Create: `skills/model-optimization/sglang-model-day0-support/scripts/validate_day0_bundle.py`

- [ ] **Step 1: Write failing validator tests**

Load the validator and define a complete fixture:

```python
validator = load_script("validate_day0_bundle")

COMPLETE_FILES = {
    "scope-contract.md": """# Day-0 Scope Contract

## Release Cut
Model and source revisions are immutable.
## Required Capabilities
Text, tools, multimodal, and PD are required.
## Out of Scope
CPU performance tuning is post-Day-0.
""",
    "architecture-gap-map.md": """# Architecture Gap Map

## Capability Classification
Model loading is day0-required.
## Evidence
- Evidence: https://github.com/sgl-project/sglang/pull/23882 | state: merged | head: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa | limitation: public support baseline only
""",
    "pr-dag.md": """# Pull Request DAG

## Dependencies
Infrastructure precedes the model spine.
## Merge Gates
Protocol and state tests gate publication.
""",
    "validation-matrix.md": """# Validation Matrix

## Risk Pairs
Speculative decoding crosses recurrent state.
## Required Lanes
TP and PD lanes are required.
""",
    "release-lock.md": """# Release Lock

## Source Revisions
All revisions are immutable.
## Artifacts
Public source and image are locked.
## Limitations
Only documented hardware is covered.
""",
    "pr-body.md": """# Public Pull Request Body

## Summary
Add the fictional model.
## Validation
Load, protocol, state, topology, and quality gates pass.
## Limitations
Only documented hardware is covered.
## Evidence
- Evidence: https://github.com/sgl-project/sglang/pull/23882 | state: merged | head: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa | limitation: public support baseline only
""",
    "follow-up-ledger.md": """# Post-Day-0 Follow-up Ledger

## Open Fixes
No release blockers remain.
## Performance Work
Kernel tuning is tracked separately.
## Experiments and Reverts
Experiments are not shipped behavior.
""",
    "sanitization-report.md": """# Sanitization Report

## Public Evidence
All evidence is public upstream evidence.
## Denylist Result
No forbidden entries were found.
""",
}


def write_complete_bundle(root: Path) -> None:
    for name, content in COMPLETE_FILES.items():
        (root / name).write_text(content, encoding="utf-8")
```

Add tests covering:

```python
def test_complete_bundle_passes(tmp_path):
    write_complete_bundle(tmp_path)
    assert validator.validate_bundle(tmp_path, {"sgl-project/sglang"}, []) == []


def test_missing_file_and_placeholder_are_reported_together(tmp_path):
    write_complete_bundle(tmp_path)
    (tmp_path / "release-lock.md").unlink()
    (tmp_path / "scope-contract.md").write_text(
        "# Day-0 Scope Contract\n\n## Release Cut\n{{MODEL_ID}}\n",
        encoding="utf-8",
    )
    findings = validator.validate_bundle(tmp_path, {"sgl-project/sglang"}, [])
    assert any("missing required file: release-lock.md" in item for item in findings)
    assert any("unresolved placeholder" in item for item in findings)


def test_private_url_absolute_path_ip_and_denylist_are_rejected(tmp_path):
    write_complete_bundle(tmp_path)
    append = "\n".join(
        (
            "",
            "https://github.com/private/example/pull/1",
            "".join(("/", "home", "/example/model")),
            ".".join(("192", "0", "2", "10")),
            "forbidden-release-token",
            "",
        )
    )
    with (tmp_path / "pr-body.md").open("a", encoding="utf-8") as handle:
        handle.write(append)
    findings = validator.validate_bundle(
        tmp_path, {"sgl-project/sglang"}, ["forbidden-release-token"]
    )
    assert any("repository is not allowed" in item for item in findings)
    assert any("absolute private path" in item for item in findings)
    assert any("IP address" in item for item in findings)
    assert any("denylist entry" in item for item in findings)


def test_open_evidence_requires_head_and_limitation(tmp_path):
    write_complete_bundle(tmp_path)
    path = tmp_path / "architecture-gap-map.md"
    path.write_text(
        path.read_text(encoding="utf-8")
        + "\n- Evidence: https://github.com/sgl-project/sglang/pull/32541 | state: open\n",
        encoding="utf-8",
    )
    findings = validator.validate_bundle(tmp_path, {"sgl-project/sglang"}, [])
    assert any("open evidence requires immutable head" in item for item in findings)
    assert any("open evidence requires limitation" in item for item in findings)
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
pytest -q tests/test_sglang_model_day0_support.py -k 'bundle or evidence or private'
```

Expected: failure because the validator module does not exist.

- [ ] **Step 3: Implement the validator**

Define:

```python
REQUIRED_FILES = {
    "scope-contract.md": ("# Day-0 Scope Contract", "## Release Cut", "## Required Capabilities", "## Out of Scope"),
    "architecture-gap-map.md": ("# Architecture Gap Map", "## Capability Classification", "## Evidence"),
    "pr-dag.md": ("# Pull Request DAG", "## Dependencies", "## Merge Gates"),
    "validation-matrix.md": ("# Validation Matrix", "## Risk Pairs", "## Required Lanes"),
    "release-lock.md": ("# Release Lock", "## Source Revisions", "## Artifacts", "## Limitations"),
    "pr-body.md": ("# Public Pull Request Body", "## Summary", "## Validation", "## Limitations", "## Evidence"),
    "follow-up-ledger.md": ("# Post-Day-0 Follow-up Ledger", "## Open Fixes", "## Performance Work", "## Experiments and Reverts"),
    "sanitization-report.md": ("# Sanitization Report", "## Public Evidence", "## Denylist Result"),
}
```

Collect all independent findings. Reject brace placeholders, `TBD`, `TODO`,
disallowed GitHub repositories, absolute private paths, IPv4 addresses,
secret-like prefixes, SSH GitHub URLs, and explicit denylist strings. Parse
each `- Evidence:` line; require `state`, a 40-hex `head`, and a non-empty
`limitation` for open evidence. Support text and JSON CLI output.

- [ ] **Step 4: Run the full targeted tests**

Run:

```bash
pytest -q tests/test_sglang_model_day0_support.py
```

Expected: all tests pass.

- [ ] **Step 5: Validate the resolved demonstration**

Run:

```bash
python3 skills/model-optimization/sglang-model-day0-support/scripts/validate_day0_bundle.py \
  docs/assets/sglang-model-day0-support-demo
```

Expected: `Day-0 bundle is valid.`

- [ ] **Step 6: Confirm unresolved templates fail**

Run:

```bash
python3 skills/model-optimization/sglang-model-day0-support/scripts/validate_day0_bundle.py \
  skills/model-optimization/sglang-model-day0-support/assets/day0-bundle
```

Expected: non-zero status with unresolved-placeholder findings.

- [ ] **Step 7: Commit scripts and tests**

Run:

```bash
git add skills/model-optimization/sglang-model-day0-support/scripts \
  tests/test_sglang_model_day0_support.py
git commit -m "feat: validate public Day-0 evidence bundles"
```

### Task 7: Complete metadata, documentation contracts, and final validation

**Files:**
- Modify: `skills/model-optimization/sglang-model-day0-support/agents/openai.yaml`
- Modify: `tests/test_sglang_model_day0_support.py`
- Modify any new files only to fix validation findings.

- [ ] **Step 1: Add documentation contract tests**

Add tests asserting:

```python
def test_skill_routes_all_references_and_stays_concise():
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert len(skill.splitlines()) < 500
    for name in (
        "day0-contract.md",
        "evidence-audit.md",
        "sanitization.md",
        "kimi-k3-case-study.md",
        "deepseek-v4-case-study.md",
    ):
        assert name in skill


def test_case_studies_use_only_public_sglang_pr_urls():
    for path in (SKILL_ROOT / "references").glob("*case-study.md"):
        text = path.read_text(encoding="utf-8")
        urls = re.findall(r"https://github\\.com/[^\\s)]+/pull/[0-9]+", text)
        assert urls
        assert all(url.startswith("https://github.com/sgl-project/sglang/pull/") for url in urls)


def test_agent_default_prompt_names_the_skill():
    metadata = (SKILL_ROOT / "agents/openai.yaml").read_text(encoding="utf-8")
    assert "$sglang-model-day0-support" in metadata
```

- [ ] **Step 2: Regenerate and inspect agent metadata**

Run:

```bash
python3 "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/generate_openai_yaml.py" \
  skills/model-optimization/sglang-model-day0-support \
  --interface 'display_name=SGLang Model Day-0 Support' \
  --interface 'short_description=Plan and audit model Day-0 support for SGLang' \
  --interface 'default_prompt=Use $sglang-model-day0-support to design and validate an auditable SGLang Day-0 support bundle for this model.'
```

Expected: quoted interface fields and no unrequested icons, colors, or tool dependencies.

- [ ] **Step 3: Run focused validation**

Run:

```bash
python3 "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py" \
  skills/model-optimization/sglang-model-day0-support
pytest -q tests/test_sglang_model_day0_support.py
python3 skills/model-optimization/sglang-model-day0-support/scripts/validate_day0_bundle.py \
  docs/assets/sglang-model-day0-support-demo
```

Expected: skill valid, all tests pass, demo valid.

- [ ] **Step 4: Run formatting and repository checks**

Run:

```bash
python3 -m py_compile \
  skills/model-optimization/sglang-model-day0-support/scripts/collect_public_pr_evidence.py \
  skills/model-optimization/sglang-model-day0-support/scripts/validate_day0_bundle.py
pre-commit run --all-files
git diff --check origin/main...HEAD
```

Expected: all commands pass. Fix only findings caused by this branch.

- [ ] **Step 5: Run the final sanitization scan**

Run:

```bash
git diff --unified=0 origin/main...HEAD -- \
  skills/model-optimization/sglang-model-day0-support \
  docs/assets/sglang-model-day0-support-demo \
  docs/superpowers/specs/2026-07-28-sglang-model-day0-support-design.md \
  docs/superpowers/plans/2026-07-28-sglang-model-day0-support.md \
tests/test_sglang_model_day0_support.py |
rg '/[U]sers/|/[h]''ome/|/[d]''ata/|git''@|dkr\\.ecr|\\b(?:[0-9]{1,3}\\.){3}[0-9]{1,3}\\b' &&
  exit 1 || true
```

Expected: no private identifier is present in added public content. Synthetic
negative-test strings must be represented through string assembly if this scan
would otherwise match them literally.

- [ ] **Step 6: Commit final validation fixes**

Run:

```bash
git add skills/model-optimization/sglang-model-day0-support \
  docs/assets/sglang-model-day0-support-demo \
  tests/test_sglang_model_day0_support.py
git commit -m "test: verify SGLang model Day-0 support skill"
```

### Task 8: Publish the focused draft pull request

**Files:**
- No new source files.

- [ ] **Step 1: Review commits and branch scope**

Run:

```bash
git status --short --branch
git log --oneline origin/main..HEAD
git diff --stat origin/main...HEAD
```

Expected: a clean branch containing only the design, plan, new skill,
demonstration, and targeted tests.

- [ ] **Step 2: Push the branch**

Run:

```bash
git push -u origin agent/add-sglang-model-day0-support
```

- [ ] **Step 3: Open a draft pull request**

Create a draft PR against `BBuf/AI-Infra-Auto-Driven-SKILLS:main` with:

```markdown
## Summary

- add an evidence-driven SGLang model Day-0 support workflow
- include public Kimi K3 and DeepSeek V4 case studies
- add copyable bundle templates, deterministic audits, and a worked demo

## Public evidence boundary

All model evidence uses public `sgl-project/sglang` PRs and public source paths.
The skill includes a fail-closed clean-room publication policy and bundle scan.

## Validation

- skill quick validation
- targeted pytest suite
- resolved demo bundle validation
- Python compile checks
- pre-commit
- staged-diff sanitization scan

## Demonstration

The fictional hybrid MoE VLM example produces a scope contract, architecture
gap map, PR DAG, risk-pair validation matrix, release lock, public PR body,
follow-up ledger, and sanitization report.
```

- [ ] **Step 4: Verify the PR**

Run:

```bash
gh pr view --repo BBuf/AI-Infra-Auto-Driven-SKILLS \
  --json number,title,state,isDraft,baseRefName,headRefName,url
```

Expected: one open draft PR with base `main` and head
`agent/add-sglang-model-day0-support`.
