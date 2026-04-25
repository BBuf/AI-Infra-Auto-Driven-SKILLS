from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISTORY_ROOT = ROOT / "model-pr-optimization-history"
SKILL_ROOT = ROOT / "skills" / "model-optimization"

ZH_START = "<!-- MODEL_PR_DIFF_AUDIT:START zh -->"
ZH_END = "<!-- MODEL_PR_DIFF_AUDIT:END zh -->"
EN_START = "<!-- MODEL_PR_DIFF_AUDIT:START en -->"
EN_END = "<!-- MODEL_PR_DIFF_AUDIT:END en -->"
REF_START = "<!-- MODEL_PR_DIFF_AUDIT:START reference -->"
REF_END = "<!-- MODEL_PR_DIFF_AUDIT:END reference -->"

PLACEHOLDER_PATTERNS = [
    r"\bTODO\b",
    r"\bTBD\b",
    r"待补",
    r"待完善",
    r"后续补",
    r"暂缺",
    r"略过",
    r"偷懒",
]


def _section(text: str, start: str, end: str) -> str:
    match = re.search(re.escape(start) + r"(.*?)" + re.escape(end), text, re.S)
    assert match, f"missing generated audit section {start}"
    return match.group(1)


def _pr_cards(section: str) -> list[str]:
    return re.findall(r"^### PR #\d+.*?(?=^### PR #\d+|\Z)", section, re.S | re.M)


def _assert_no_placeholders(path: Path, section: str) -> None:
    for pattern in PLACEHOLDER_PATTERNS:
        assert not re.search(pattern, section, re.I), f"{path} contains placeholder pattern {pattern!r}"
    assert "GitHub API 返回" not in section, f"{path} contains unresolved PR lookup"
    assert "GitHub API returned" not in section, f"{path} contains unresolved PR lookup"


def _assert_zh_cards(path: Path, section: str) -> None:
    if "公开 PR 检索结论" in section:
        assert "未确认到可归入该模型支持或优化主线的公开 PR" in section
        return
    cards = _pr_cards(section)
    assert cards, f"{path} has no PR diff cards"
    for card in cards:
        for required in [
            "状态/时间",
            "代码 diff 已读范围",
            "代码 diff 细节",
            "支持/优化点判断",
            "风险与验证",
        ]:
            assert required in card, f"{path} card missing {required}"
        has_file_digest = re.search(r"`[^`]+`\s+\w+\s+\+\d+/-\d+", card)
        has_empty_diff_note = "GitHub 未返回文件级 patch" in card
        assert has_file_digest or has_empty_diff_note, f"{path} card lacks file diff digest"


def _assert_en_cards(path: Path, section: str) -> None:
    if "Public PR search conclusion" in section:
        assert "No public PR was confirmed" in section
        return
    cards = _pr_cards(section)
    assert cards, f"{path} has no PR diff cards"
    for card in cards:
        for required in [
            "Status/date",
            "Diff scope read",
            "Code diff details",
            "Optimization/support interpretation",
            "Risk and verification",
        ]:
            assert required in card, f"{path} card missing {required}"
        has_file_digest = re.search(r"`[^`]+`\s+\w+\s+\+\d+/-\d+", card)
        has_empty_diff_note = "No patch file list returned" in card
        assert has_file_digest or has_empty_diff_note, f"{path} card lacks file diff digest"


def test_history_docs_have_audited_pr_diff_cards() -> None:
    history_docs = sorted(HISTORY_ROOT.glob("*/**/README.*.md"))
    history_docs = [p for p in history_docs if p.name in {"README.zh.md", "README.en.md"}]
    assert history_docs, "no model PR history docs found"

    for path in history_docs:
        text = path.read_text(encoding="utf-8")
        if path.name == "README.zh.md":
            section = _section(text, ZH_START, ZH_END)
            _assert_no_placeholders(path, section)
            _assert_zh_cards(path, section)
        else:
            section = _section(text, EN_START, EN_END)
            _assert_no_placeholders(path, section)
            _assert_en_cards(path, section)


def test_skill_pr_history_references_have_audited_pr_diff_cards() -> None:
    references = sorted(SKILL_ROOT.glob("*/*/references/pr-history.md"))
    assert references, "no model optimization skill PR references found"

    for path in references:
        section = _section(path.read_text(encoding="utf-8"), REF_START, REF_END)
        _assert_no_placeholders(path, section)
        if "No public PR confirmed" in section:
            continue
        _assert_en_cards(path, section)


def test_model_optimization_skill_entries_link_to_diff_dossier_rule() -> None:
    skill_docs = sorted(SKILL_ROOT.glob("*/*/SKILL.md"))
    assert skill_docs, "no model optimization skill entry docs found"

    for path in skill_docs:
        text = path.read_text(encoding="utf-8")
        assert "references/pr-history.md" in text, f"{path} must link its audited PR history"
        assert "model-pr-diff-dossier" in text, f"{path} must point to the diff dossier standard"
        assert "not only PR titles" in text, f"{path} must reject title-only PR summaries"
