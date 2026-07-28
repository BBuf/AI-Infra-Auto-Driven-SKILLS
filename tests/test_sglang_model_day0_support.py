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
    record = collector.build_record(
        PR_PAYLOAD,
        FILE_PAYLOAD,
        captured_at="2026-07-28T00:00:00Z",
    )
    assert record["state"] == "merged"
    assert record["head_sha"] == "a" * 40
    assert record["files"][0]["filename"] == "python/sglang/srt/models/example.py"
    assert "motivation" not in record
    assert "implementation" not in record
    assert "performance" not in record


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
    findings = validator.validate_bundle(
        tmp_path,
        {"sgl-project/sglang"},
        [],
    )
    assert any(
        "missing required file: release-lock.md" in item for item in findings
    )
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
        tmp_path,
        {"sgl-project/sglang"},
        ["forbidden-release-token"],
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
        + "\n- Evidence: https://github.com/sgl-project/sglang/pull/32541"
        " | state: open\n",
        encoding="utf-8",
    )
    findings = validator.validate_bundle(
        tmp_path,
        {"sgl-project/sglang"},
        [],
    )
    assert any("open evidence requires immutable head" in item for item in findings)
    assert any("open evidence requires limitation" in item for item in findings)


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
        urls = re.findall(
            r"https://github\.com/[^\s)]+/pull/[0-9]+",
            text,
        )
        assert urls
        assert all(
            url.startswith("https://github.com/sgl-project/sglang/pull/")
            for url in urls
        )


def test_agent_default_prompt_names_the_skill():
    metadata = (SKILL_ROOT / "agents/openai.yaml").read_text(encoding="utf-8")
    assert "$sglang-model-day0-support" in metadata
