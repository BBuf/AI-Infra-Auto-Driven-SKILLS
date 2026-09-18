from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "skills" / "sglang-humanize-review"
SCRIPTS = SKILL_ROOT / "scripts"
CORPUS = SKILL_ROOT / "references" / "sglang-review-corpus.jsonl.gz"
METADATA = SKILL_ROOT / "references" / "sglang-review-corpus.metadata.json"
QUERY_SCRIPT = SCRIPTS / "query_sglang_review_corpus.py"
SUMMARIZE_SCRIPT = SCRIPTS / "summarize_sglang_review_corpus.py"

sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "sglang_review_summarizer", SUMMARIZE_SCRIPT
)
assert SPEC and SPEC.loader
SUMMARIZER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUMMARIZER
SPEC.loader.exec_module(SUMMARIZER)


def _review_row(number: int) -> dict[str, object]:
    return {
        "row_type": "inline_review_thread",
        "thread_id": f"thread-{number}",
        "path": f"python/sglang/file_{number}.py",
        "diff_hunk": "@@ -1 +1 @@\n-old\n+new",
        "categories": ["correctness"],
        "human_reviewer_comment_count": 1,
        "agent_reviewer_comment_count": 0,
        "pull_request": {
            "number": number,
            "title": f"Needle change {number}",
            "created_at": "2026-07-01T00:00:00Z",
        },
        "comments": [
            {
                "kind": "inline_review_comment",
                "body": f"needle review {number}",
                "html_url": f"https://example.com/{number}",
                "author": {"login": "human", "is_agent": False},
            }
        ],
    }


def test_summarizer_bounds_digest_payloads_and_streams_jsonl(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "reviews.jsonl.gz"
    with gzip.open(corpus, "wt", encoding="utf-8") as handle:
        for number in range(1, 7):
            handle.write(json.dumps(_review_row(number)) + "\n")

    args = argparse.Namespace(
        corpus=corpus,
        path=[],
        query=["needle"],
        category=[],
        kind="",
        reviewer="",
        batch_size=2,
        top=2,
        include_agent_reviewers=False,
        excerpt_chars=320,
        format="digest",
    )
    result = SUMMARIZER.sweep_corpus(args)

    assert result.total_scanned == 6
    assert result.matched_count == 6
    assert len(result.top_matches) == 2

    streamed = subprocess.run(
        [
            "python3",
            str(SUMMARIZE_SCRIPT),
            "--corpus",
            str(corpus),
            "--query",
            "needle",
            "--batch-size",
            "2",
            "--top",
            "2",
            "--format",
            "jsonl",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    streamed_rows = [json.loads(line) for line in streamed.stdout.splitlines()]
    assert [row["pull_request"]["number"] for row in streamed_rows] == list(range(1, 7))


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


def test_summarize_script_scans_whole_corpus_and_aggregates() -> None:
    result = subprocess.run(
        [
            "python3",
            str(SUMMARIZE_SCRIPT),
            "--path",
            "python/sglang/srt/speculative",
            "--query",
            "eagle",
            "--top",
            "2",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    # Exhaustive coverage line over the full corpus, plus the aggregate and the
    # ranked excerpts that the review synthesis is built from.
    assert "Scanned" in result.stdout and "matched" in result.stdout
    assert "## Aggregate over ALL matches" in result.stdout
    assert "most relevant review threads" in result.stdout
    assert "_relevance:" in result.stdout


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
