#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import re
import textwrap
from pathlib import Path
from typing import Any, Iterable

DEFAULT_CORPUS = (
    Path(__file__).resolve().parents[1]
    / "references"
    / "sglang-review-corpus-2024-2025.jsonl.gz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search the SGLang human review corpus."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--query", default="", help="Case-insensitive text query.")
    parser.add_argument("--path", default="", help="Path substring filter.")
    parser.add_argument("--category", default="", help="Category label filter.")
    parser.add_argument(
        "--kind",
        default="",
        help=(
            "Thread/comment kind filter, for example inline_review_thread, "
            "pr_conversation, or review_submission."
        ),
    )
    parser.add_argument("--pr", type=int, default=0, help="Pull request number filter.")
    parser.add_argument("--reviewer", default="", help="Reviewer login filter.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--include-agent-reviewers",
        action="store_true",
        help="Include threads with only bot or coding-agent reviewer comments.",
    )
    parser.add_argument(
        "--show-discussion",
        action="store_true",
        help="Show all comments in a matching thread, including agent comments.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "jsonl"),
        default="markdown",
    )
    return parser.parse_args()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    with open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def thread_text(thread: dict[str, Any]) -> str:
    parts = [
        thread.get("row_type", ""),
        thread.get("path", ""),
        thread.get("diff_hunk", ""),
        thread.get("pull_request", {}).get("title", ""),
        " ".join(thread.get("categories", [])),
    ]
    for comment in thread.get("comments", []):
        parts.extend(
            [
                comment.get("kind", ""),
                comment.get("state", ""),
                comment.get("body", ""),
            ]
        )
    return "\n".join(parts)


def row_type_for_thread(thread: dict[str, Any]) -> str:
    row_type = thread.get("row_type")
    if row_type:
        return str(row_type)
    collection = str(thread.get("source", {}).get("collection", ""))
    if collection == "github_pull_review_comments":
        return "inline_review_thread"
    return collection or "inline_review_thread"


def matches(thread: dict[str, Any], args: argparse.Namespace) -> bool:
    if (
        not args.include_agent_reviewers
        and thread.get("human_reviewer_comment_count", 0) == 0
    ):
        return False
    if args.path and args.path.lower() not in thread.get("path", "").lower():
        return False
    if args.category and args.category not in thread.get("categories", []):
        return False
    if args.kind:
        kind = args.kind.lower()
        row_type = row_type_for_thread(thread)
        comment_kinds = [
            str(comment.get("kind", "")) for comment in thread.get("comments", [])
        ]
        if kind != row_type.lower() and kind not in {
            item.lower() for item in comment_kinds
        }:
            return False
    if args.pr and args.pr != int(thread.get("pull_request", {}).get("number", 0)):
        return False
    if args.reviewer:
        reviewer = args.reviewer.lower()
        if not any(
            reviewer == str(comment.get("author", {}).get("login", "")).lower()
            for comment in thread.get("comments", [])
        ):
            return False
    if args.query:
        return (
            re.search(re.escape(args.query), thread_text(thread), re.IGNORECASE)
            is not None
        )
    return True


def clip(text: str, width: int = 520) -> str:
    clean = text.strip()
    if len(clean) <= width:
        return clean
    return clean[: width - 3].rstrip() + "..."


def render_markdown(thread: dict[str, Any], show_discussion: bool) -> str:
    pr = thread["pull_request"]
    url = ""
    if thread.get("comments"):
        url = thread["comments"][0].get("html_url") or ""
    comments = thread.get("comments", [])
    if not show_discussion:
        comments = [
            comment
            for comment in comments
            if not comment.get("author", {}).get("is_agent", False)
        ]

    lines = [
        f"### PR #{pr['number']}: {pr['title']}",
        "",
        f"- Type: `{row_type_for_thread(thread)}`",
        f"- Path: `{thread.get('path', '') or '<conversation>'}`",
        f"- Categories: `{', '.join(thread.get('categories', []))}`",
        f"- Link: {url}",
        "",
    ]
    diff_hunk = thread.get("diff_hunk", "")
    if diff_hunk:
        lines.extend(
            [
                "```diff",
                clip(diff_hunk, 1200),
                "```",
                "",
            ]
        )
    for comment in comments:
        author = comment.get("author", {})
        marker = "agent" if author.get("is_agent") else "human"
        kind = comment.get("kind") or thread.get("row_type") or "inline_review_comment"
        state = f", {comment.get('state')}" if comment.get("state") else ""
        body = textwrap.indent(clip(comment.get("body", ""), 900), "> ")
        lines.extend(
            [
                f"**{author.get('login', 'unknown')}** ({marker}, {kind}{state}, {comment.get('created_at', '')})",
                "",
                body or "> <empty body>",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    found = 0
    for thread in iter_rows(args.corpus):
        if not matches(thread, args):
            continue
        found += 1
        if args.format == "jsonl":
            print(json.dumps(thread, ensure_ascii=False, sort_keys=True))
        else:
            print(render_markdown(thread, args.show_discussion))
            print()
        if found >= args.limit:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
