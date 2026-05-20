from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "skills" / "sglang-humanize-review"
CORPUS = SKILL_ROOT / "references" / "sglang-review-corpus-2024-2025.jsonl.gz"
METADATA = SKILL_ROOT / "references" / "sglang-review-corpus-2024-2025.metadata.json"
QUERY_SCRIPT = SKILL_ROOT / "scripts" / "query_sglang_review_corpus.py"


def test_skill_points_to_corpus_and_review_workflow() -> None:
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")

    assert "2024" in text
    assert "2025" in text
    assert "corpus-summary.md" in text
    assert "query_sglang_review_corpus.py" in text
    assert "diff_hunk" in text
    assert "original comment language" in text
    assert "Findings first" in text
    assert "SGLang Review Heuristics" in text


def test_metadata_covers_2024_2025_human_review_corpus() -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    hints = dict(metadata["comment_language_hints"])

    assert metadata["source_years"] == [2024, 2025]
    assert metadata["pull_request_stats"]["included_human_prs"] >= 11000
    assert metadata["thread_count"] >= 10000
    assert metadata["human_reviewer_comment_count"] == metadata["comment_count"]
    assert metadata["agent_reviewer_comment_count"] == 0
    assert hints["en_or_ascii"] > 0
    assert hints["zh_or_cjk"] > 0
    assert hints["non_ascii_other"] > 0


def test_corpus_rows_preserve_code_context_and_comments() -> None:
    with gzip.open(CORPUS, "rt", encoding="utf-8") as handle:
        row = json.loads(next(handle))

    assert row["pull_request"]["created_at"].startswith(("2024", "2025"))
    assert row["pull_request"]["author"]["is_agent"] is False
    assert row["diff_hunk"].startswith("@@")
    assert row["path"]
    assert row["comments"]
    assert row["comments"][0]["body"]
    assert row["comments"][0]["author"]["is_agent"] is False
    assert row["categories"]


def test_query_script_returns_markdown_with_diff_context() -> None:
    result = subprocess.run(
        [
            "python3",
            str(QUERY_SCRIPT),
            "--query",
            "server_args",
            "--limit",
            "1",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    assert "### PR #" in result.stdout
    assert "```diff" in result.stdout
    assert "**" in result.stdout
