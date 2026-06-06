from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "skills" / "sglang-humanize-review"
CORPUS = SKILL_ROOT / "references" / "sglang-review-corpus.jsonl.gz"
METADATA = SKILL_ROOT / "references" / "sglang-review-corpus.metadata.json"
QUERY_SCRIPT = SKILL_ROOT / "scripts" / "query_sglang_review_corpus.py"


def test_skill_points_to_corpus_and_review_workflow() -> None:
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")

    assert "2026" in text
    assert "project start" in text
    assert "corpus-summary.md" in text
    assert "query_sglang_review_corpus.py" in text
    assert "diff_hunk" in text
    assert "original comment language" in text
    assert "pr_conversation" in text
    assert "review_submission" in text
    assert "model-pr-optimization-history" in text
    assert "Findings first" in text
    assert "SGLang Review Heuristics" in text


def test_metadata_covers_full_human_review_episode_corpus() -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    hints = dict(metadata["comment_language_hints"])
    thread_types = dict(metadata.get("threads_by_type", []))
    event_kinds = dict(metadata.get("events_by_kind", []))

    assert metadata["schema_version"] >= 2
    assert metadata["source_years"] == [2024, 2026]
    assert metadata["pull_request_stats"]["included_human_prs"] >= 11000
    assert metadata["thread_count"] >= 10000
    assert metadata["human_reviewer_comment_count"] == metadata["comment_count"]
    assert metadata["agent_reviewer_comment_count"] == 0
    assert thread_types["inline_review_thread"] >= 10000
    assert thread_types["pr_conversation"] > 0
    assert thread_types["review_submission"] > 0
    assert event_kinds["inline_review_comment"] > 0
    assert event_kinds["pr_conversation"] > 0
    assert event_kinds["review_submission"] > 0
    assert hints["en_or_ascii"] > 0
    assert hints["zh_or_cjk"] > 0
    assert hints["non_ascii_other"] > 0


def test_corpus_rows_preserve_code_context_and_comments() -> None:
    with gzip.open(CORPUS, "rt", encoding="utf-8") as handle:
        row = json.loads(next(handle))

    assert row["pull_request"]["created_at"].startswith(("2024", "2025", "2026"))
    assert row["pull_request"]["author"]["is_agent"] is False
    assert row["row_type"] in {
        "inline_review_thread",
        "pr_conversation",
        "review_submission",
    }
    if row["row_type"] == "inline_review_thread":
        assert row["diff_hunk"].startswith("@@")
        assert row["path"]
    assert row["comments"]
    assert row["comments"][0]["body"]
    assert row["comments"][0]["author"]["is_agent"] is False
    assert row["comments"][0]["kind"]
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


def test_query_script_can_filter_conversation_and_review_submission() -> None:
    for kind in ("pr_conversation", "review_submission"):
        result = subprocess.run(
            [
                "python3",
                str(QUERY_SCRIPT),
                "--kind",
                kind,
                "--limit",
                "1",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        assert f"- Type: `{kind}`" in result.stdout
