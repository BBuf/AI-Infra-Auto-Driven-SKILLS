from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_uses_current_claude_and_codex_launch_commands() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "claude --model opus --permission-mode auto" in readme
    assert "codex --sandbox danger-full-access --ask-for-approval never" in readme
    assert "Opus 4.8" not in readme
    assert "codex --yolo" not in readme
    assert "`opus`" in readme and "current Opus" in readme
    assert "bypassPermissions" in readme and "isolated" in readme
    assert "core_skills-11" in readme


def test_marketplace_has_top_level_description() -> None:
    marketplace = json.loads(
        (ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8")
    )

    assert marketplace["description"]
    assert "LLM serving" in marketplace["description"]
    assert marketplace["plugins"][0]["description"]


def test_precommit_versions_are_current_verified_tags() -> None:
    config = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    for revision in [
        "rev: v6.0.0",
        "rev: 8.0.1",
        "rev: v0.16.0",
        "rev: 26.5.1",
        "rev: v2.4.3",
        "rev: v22.1.8",
        "rev: lychee-v0.24.2",
    ]:
        assert revision in config


def test_lint_workflow_uses_current_verified_actions() -> None:
    workflow = (ROOT / ".github" / "workflows" / "lint.yml").read_text(encoding="utf-8")

    assert "actions/checkout@v7" in workflow
    assert "actions/setup-python@v7" in workflow
    assert (
        "lycheeverse/lychee-action@e7477775783ea5526144ba13e8db5eec57747ce8" in workflow
    )
    assert "# v2.9.0" in workflow
    assert "DoozyX/clang-format-lint-action@v0.20" in workflow
    assert "clangFormatVersion: 22" in workflow


def test_refresh_prompt_is_pr_agnostic_and_preserves_evidence_gates() -> None:
    prompt = (ROOT / "update_prompt.md").read_text(encoding="utf-8")

    assert "PR 72" not in prompt
    assert "pr72" not in prompt
    for required in [
        "open-pr-watch.md",
        "5 个不同规模模型",
        "MiniMax-M3",
        "llm-torch-profiler-analysis",
        "ncu --set basic",
        "本地和远端测试都通过",
        "CI 状态已核对",
    ]:
        assert required in prompt
