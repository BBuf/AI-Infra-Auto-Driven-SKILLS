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
    assert "core_skills-12" in readme
    assert "After reload, the 12 skills appear" in readme


def test_sglang_day0_skill_is_discoverable_and_installable() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    model_index = (ROOT / "skills" / "model-optimization" / "README.md").read_text(
        encoding="utf-8"
    )
    skill_path = "skills/model-optimization/sglang-model-day0-support"

    assert "[`sglang-model-day0-support`]" in readme
    assert (
        f'ln -s "$PWD/{skill_path}" ' "~/.claude/skills/sglang-model-day0-support"
    ) in readme
    assert (
        f"cp -R {skill_path} " "<agent-skill-dir>/sglang-model-day0-support"
    ) in readme
    assert "└── sglang-model-day0-support/" in readme
    assert "`sglang-model-day0-support/`" in model_index


def test_marketplace_has_top_level_description() -> None:
    marketplace = json.loads(
        (ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8")
    )
    plugin = json.loads(
        (ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
    )

    assert marketplace["description"]
    assert "LLM serving" in marketplace["description"]
    assert marketplace["plugins"][0]["description"]
    assert marketplace["plugins"][0]["version"] == "0.4.0"
    assert plugin["version"] == "0.4.0"
    assert marketplace["plugins"][0]["version"] == plugin["version"]
    assert "Day-0" in marketplace["description"]
    assert "Day-0" in plugin["description"]


def test_retired_diffusion_skill_names_are_absent_from_current_tree() -> None:
    retired_names = {
        "-".join(("sglang", "diffusion", "auto", "optimize")),
        "-".join(("sol", "engine", "sglang", "diffusion")),
    }
    text_suffixes = {".json", ".md", ".py", ".toml", ".yaml", ".yml"}
    for path in ROOT.rglob("*"):
        if (
            path.is_file()
            and path.suffix in text_suffixes
            and ".git" not in path.parts
        ):
            text = path.read_text(encoding="utf-8")
            for retired_name in retired_names:
                assert retired_name not in text


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
    assert "216.114.73.196" not in prompt
    assert "cirrascale-gpuc5a6" not in prompt
    assert "radix machines mine --json" in prompt
    assert "不要把历史机器名、IP 或已过期" in prompt
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
