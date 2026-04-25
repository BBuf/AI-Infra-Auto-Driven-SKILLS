from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISTORY_ROOT = ROOT / "model-pr-optimization-history"
SKILL_ROOT = ROOT / "skills" / "model-optimization"

DIFFUSION_SLUGS = {
    "ltx23-hq",
    "qwen-image",
    "z-image-turbo",
    "sglang-ltx23-hq-optimization",
    "sglang-qwen-image-optimization",
    "sglang-z-image-turbo-optimization",
    "vllm-ltx23-hq-optimization",
    "vllm-qwen-image-optimization",
    "vllm-z-image-turbo-optimization",
}

PLACEHOLDER_PATTERNS = [
    r"待补",
    r"待完善",
    r"后续补",
    r"暂缺",
    r"略过",
    r"偷懒",
    r"unavailable PR",
    r"GitHub API 返回",
    r"GitHub API returned",
]

ZH_REQUIRED_SECTIONS = [
    "## 文档口径",
    "## 模型实现文件覆盖",
    "## PR 覆盖总览",
    "## 时间线",
    "## 逐 PR diff 审计卡",
]

EN_REQUIRED_SECTIONS = [
    "## Scope",
    "## Implementation File Coverage",
    "## PR Coverage Summary",
    "## Timeline",
    "## Per-PR Diff Audit Cards",
]

REF_REQUIRED_SECTIONS = [
    "## Implementation File Coverage",
    "## Timeline",
    "## Per-PR Diff Audit Cards",
]


def _pr_cards(text: str) -> list[str]:
    return re.findall(r"^### PR #\d+.*?(?=^### PR #\d+|\Z)", text, re.S | re.M)


def _assert_no_placeholders(path: Path, text: str) -> None:
    for pattern in PLACEHOLDER_PATTERNS:
        assert not re.search(pattern, text, re.I), f"{path} contains placeholder pattern {pattern!r}"


def _assert_zh_cards(path: Path, text: str) -> None:
    if "### 公开 PR 检索结论" in text:
        assert "未确认到可归入" in text
        return
    cards = _pr_cards(text)
    assert cards, f"{path} has no PR diff cards"
    for card in cards:
        for required in [
            "链接:",
            "状态/时间:",
            "反查来源:",
            "代码 diff 已读范围:",
            "动机:",
            "实现要点:",
            "代码 diff 细节:",
            "关键代码摘录:",
            "已读文件:",
            "验证与风险:",
        ]:
            assert required in card, f"{path} card missing {required}"
        assert "```diff" in card, f"{path} card lacks a diff excerpt"


def _assert_en_cards(path: Path, text: str) -> None:
    if "### Public PR search conclusion" in text:
        assert "No public PR was confirmed" in text
        return
    cards = _pr_cards(text)
    assert cards, f"{path} has no PR diff cards"
    for card in cards:
        for required in [
            "Link:",
            "Status/date:",
            "Trace source:",
            "Diff scope read:",
            "Motivation:",
            "Key implementation:",
            "Code diff details:",
            "Key code excerpts:",
            "Reviewed files:",
            "Risk and verification:",
        ]:
            assert required in card, f"{path} card missing {required}"
        assert "```diff" in card, f"{path} card lacks a diff excerpt"


def test_diffusion_model_families_are_removed() -> None:
    for slug in DIFFUSION_SLUGS:
        assert not (HISTORY_ROOT / "sglang" / slug).exists()
        assert not (HISTORY_ROOT / "vllm" / slug).exists()
        assert not (SKILL_ROOT / "sglang" / slug).exists()
        assert not (SKILL_ROOT / "vllm" / slug).exists()

    index_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            HISTORY_ROOT / "sglang" / "README.md",
            HISTORY_ROOT / "vllm" / "README.md",
            SKILL_ROOT / "sglang" / "README.md",
            SKILL_ROOT / "vllm" / "README.md",
        ]
    )
    for slug in DIFFUSION_SLUGS:
        assert slug not in index_text


def test_history_docs_share_the_git_traced_diff_card_format() -> None:
    history_docs = sorted(HISTORY_ROOT.glob("*/**/README.*.md"))
    history_docs = [p for p in history_docs if p.name in {"README.zh.md", "README.en.md"}]
    assert history_docs, "no model PR history docs found"

    for path in history_docs:
        text = path.read_text(encoding="utf-8")
        _assert_no_placeholders(path, text)
        assert "GitHub Pull Request files API" in text, f"{path} must cite the diff source"
        assert "git log --name-only -- <model-files>" in text, f"{path} must cite the git trace command"
        if path.name == "README.zh.md":
            for required in ZH_REQUIRED_SECTIONS:
                assert required in text, f"{path} missing {required}"
            _assert_zh_cards(path, text)
        else:
            for required in EN_REQUIRED_SECTIONS:
                assert required in text, f"{path} missing {required}"
            _assert_en_cards(path, text)


def test_skill_pr_history_references_share_the_same_card_format() -> None:
    references = sorted(SKILL_ROOT.glob("*/*/references/pr-history.md"))
    assert references, "no model optimization skill PR references found"

    for path in references:
        text = path.read_text(encoding="utf-8")
        _assert_no_placeholders(path, text)
        assert "GitHub Pull Request files API" in text, f"{path} must cite the diff source"
        assert "not only PR titles" in text, f"{path} must reject title-only summaries"
        for required in REF_REQUIRED_SECTIONS:
            assert required in text, f"{path} missing {required}"
        _assert_en_cards(path, text)


def test_model_optimization_skill_entries_link_to_diff_dossier_rule() -> None:
    skill_docs = sorted(SKILL_ROOT.glob("*/*/SKILL.md"))
    assert skill_docs, "no model optimization skill entry docs found"

    for path in skill_docs:
        text = path.read_text(encoding="utf-8")
        assert "references/pr-history.md" in text, f"{path} must link its audited PR history"
        assert "model-pr-diff-dossier" in text, f"{path} must point to the diff dossier standard"
        assert "not only PR titles" in text, f"{path} must reject title-only PR summaries"
