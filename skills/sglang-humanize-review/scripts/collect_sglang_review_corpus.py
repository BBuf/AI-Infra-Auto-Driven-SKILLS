#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import http.client
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

API_ROOT = "https://api.github.com"
DEFAULT_REPO = "sgl-project/sglang"
DEFAULT_OUT_NAME = "sglang-review-corpus.jsonl.gz"
DEFAULT_METADATA_NAME = "sglang-review-corpus.metadata.json"

AGENT_LOGIN_PATTERNS = [
    r"\[bot\]$",
    r"^dependabot",
    r"^renovate",
    r"^github-actions",
    r"^pre-commit-ci",
    r"^copilot",
    r"^claude",
    r"^codex",
    r"^cursor",
    r"^devin",
    r"^sweep",
    r"^greptile",
    r"^openhands",
    r"^aider",
    r"^codeium",
    r"^qodo",
    r"^coderabbit",
]

CATEGORY_PATTERNS = {
    "correctness": [
        r"\bbug\b",
        r"\bwrong\b",
        r"\berror\b",
        r"\bfix\b",
        r"\bcrash\b",
        r"\bfail",
        r"\bassert",
        r"\bcorner case\b",
        r"\bedge case\b",
        r"not correct",
        r"incorrect",
    ],
    "tests-ci": [
        r"\btest",
        r"\bpytest\b",
        r"\bci\b",
        r"\bunit test\b",
        r"\bintegration",
        r"\bcoverage\b",
        r"\bbenchmark",
        r"\bworkflow",
    ],
    "performance": [
        r"\bperf",
        r"\blatency\b",
        r"\bthroughput\b",
        r"\bslow",
        r"\bfast",
        r"\bbenchmark",
        r"\bprofile",
        r"\boverhead\b",
        r"\boptimiz",
    ],
    "gpu-kernel": [
        r"\bcuda\b",
        r"\btriton\b",
        r"\bkernel\b",
        r"\bcutlass\b",
        r"\bflashinfer\b",
        r"\bflashattention\b",
        r"\bsm\d+",
        r"\bwarp\b",
    ],
    "distributed-concurrency": [
        r"\btp\b",
        r"\bdp\b",
        r"\bpp\b",
        r"\bep\b",
        r"\bnccl\b",
        r"\brank\b",
        r"\bworker\b",
        r"\bthread\b",
        r"\blocking\b",
        r"\brace\b",
        r"\bhang\b",
        r"\bdeadlock\b",
    ],
    "memory-cache": [
        r"\bmemory\b",
        r"\boom\b",
        r"\bkv\b",
        r"\bcache\b",
        r"\bpage\b",
        r"\bpool\b",
        r"\balloc",
        r"\bfree\b",
        r"\bgpu memory\b",
    ],
    "api-compat": [
        r"\bapi\b",
        r"\bopenai\b",
        r"\bcompatible\b",
        r"\bbackward",
        r"\bserver_args\b",
        r"\barg\b",
        r"\bcli\b",
        r"\bendpoint\b",
        r"\brequest\b",
        r"\bresponse\b",
    ],
    "models-quant": [
        r"\bmodel\b",
        r"\btokenizer\b",
        r"\bfp8\b",
        r"\bint4\b",
        r"\bquant",
        r"\bmoe\b",
        r"\beagle\b",
        r"\bspeculative\b",
        r"\battention\b",
    ],
    "docs-examples": [
        r"\breadme\b",
        r"\bdoc",
        r"\bexample",
        r"\btutorial",
        r"\bcomment",
        r"\bguide",
        r"\bnotebook",
    ],
    "style-maintainability": [
        r"\bstyle\b",
        r"\bnit",
        r"\bcleanup\b",
        r"\brefactor\b",
        r"\brename\b",
        r"\bduplicate\b",
        r"\bsimpl",
        r"\bmaintain",
    ],
    "build-deps": [
        r"\bbuild\b",
        r"\bcmake\b",
        r"\bsetup\b",
        r"\binstall\b",
        r"\bdependency\b",
        r"\bpackage\b",
        r"\bversion\b",
        r"\bdocker\b",
    ],
    "observability": [
        r"\blog\b",
        r"\bmetric",
        r"\bprometheus\b",
        r"\btrace\b",
        r"\bdebug\b",
        r"\bwarning\b",
        r"\bmessage\b",
    ],
}

EXTENSION_LANGUAGE = {
    ".py": "python",
    ".pyi": "python",
    ".cu": "cuda",
    ".cuh": "cuda",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".rs": "rust",
    ".go": "go",
    ".sh": "shell",
    ".bash": "shell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".md": "markdown",
    ".rst": "rst",
    ".toml": "toml",
    ".txt": "text",
    ".ipynb": "notebook",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect human SGLang pull-request review discussion episodes."
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--from-beginning",
        action="store_true",
        help=(
            "Use the repository's first PR as the start of the window. "
            "For sgl-project/sglang this currently resolves to 2024-01-08."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "references",
    )
    parser.add_argument("--out-name", default=DEFAULT_OUT_NAME)
    parser.add_argument("--metadata-name", default=DEFAULT_METADATA_NAME)
    parser.add_argument(
        "--include-agent-reviewers",
        action="store_true",
        help="Keep comments from bot or coding-agent accounts in the corpus.",
    )
    parser.add_argument(
        "--include-inline-review-comments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include inline pull-review comment threads.",
    )
    parser.add_argument(
        "--include-pr-conversations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include top-level PR conversation comments.",
    )
    parser.add_argument(
        "--include-review-submissions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include review submission bodies such as COMMENT and REQUEST_CHANGES.",
    )
    parser.add_argument(
        "--max-comment-pages",
        type=int,
        default=None,
        help="Debug limit for the global review-comments pagination.",
    )
    parser.add_argument(
        "--max-issue-comment-pages",
        type=int,
        default=None,
        help=(
            "Debug limit for PR conversation GraphQL batches. Kept under the "
            "old name for compatibility with earlier collector commands."
        ),
    )
    parser.add_argument(
        "--max-pr-pages",
        type=int,
        default=None,
        help="Debug limit for pull-request pagination.",
    )
    parser.add_argument(
        "--max-review-submission-batches",
        type=int,
        default=None,
        help="Debug limit for GraphQL review-submission batches.",
    )
    parser.add_argument(
        "--graphql-batch-size",
        type=int,
        default=25,
        help="Pull-request node IDs per GraphQL review-submission request.",
    )
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    return parser.parse_args()


def gh_token_from_cli() -> str | None:
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    token = result.stdout.strip()
    return token or None


def parse_github_dt(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def iso_utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def is_agent_user(user: dict[str, Any] | None) -> bool:
    if not user:
        return True
    login = str(user.get("login") or "").lower()
    if str(user.get("type") or "").lower() == "bot":
        return True
    return any(re.search(pattern, login) for pattern in AGENT_LOGIN_PATTERNS)


_ACTIVE_TOKEN: str | None = None


def set_active_token(token: str | None) -> None:
    global _ACTIVE_TOKEN
    _ACTIVE_TOKEN = token


def refresh_active_token(stale: str | None) -> str | None:
    """Re-read the GitHub token from the gh CLI after a 401.

    Long collection runs can outlive a rotated OAuth token; pulling a fresh one
    from `gh auth token` lets the run continue instead of dying mid-stream.
    """
    new = gh_token_from_cli()
    if new and new != stale:
        set_active_token(new)
        print("refreshed GitHub token from gh after 401", file=sys.stderr)
        return new
    return None


def api_get(
    url: str, token: str | None, retry_count: int = 12
) -> tuple[Any, dict[str, str]]:
    if not url.startswith("http"):
        url = f"{API_ROOT}{url}"

    token = _ACTIVE_TOKEN or token

    def build_request() -> urllib.request.Request:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "sglang-humanize-review-corpus",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return urllib.request.Request(url, headers=headers)

    request = build_request()
    for attempt in range(retry_count):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read().decode("utf-8")
                return json.loads(payload), {
                    k.lower(): v for k, v in response.headers.items()
                }
        except urllib.error.HTTPError as exc:
            remaining = exc.headers.get("X-RateLimit-Remaining")
            reset_at = exc.headers.get("X-RateLimit-Reset")
            if exc.code in {403, 429} and remaining == "0" and reset_at:
                sleep_s = max(int(reset_at) - int(time.time()) + 5, 5)
                print(f"rate limited; sleeping {sleep_s}s", file=sys.stderr)
                time.sleep(sleep_s)
                continue
            if exc.code in {401, 403} and attempt + 1 < retry_count:
                new = refresh_active_token(token)
                if new:
                    token = new
                    request = build_request()
                    continue
            if exc.code >= 500 and attempt + 1 < retry_count:
                time.sleep(2**attempt)
                continue
            raise
        except (http.client.HTTPException, OSError):
            # OSError covers ConnectionResetError / BrokenPipe / socket timeouts;
            # HTTPException covers IncompleteRead / RemoteDisconnected. urllib does
            # not always wrap these in URLError, so catch them directly.
            if attempt + 1 >= retry_count:
                raise
            time.sleep(min(2**attempt, 60))
    raise RuntimeError(f"failed to fetch {url}")


def api_graphql(
    query: str,
    variables: dict[str, Any],
    token: str | None,
    retry_count: int = 12,
) -> tuple[Any, dict[str, str]]:
    token = _ACTIVE_TOKEN or token
    if not token:
        raise RuntimeError("GraphQL review collection requires a GitHub token")

    payload = json.dumps(
        {"query": query, "variables": variables}, ensure_ascii=False
    ).encode("utf-8")

    def build_request() -> urllib.request.Request:
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "sglang-humanize-review-corpus",
        }
        return urllib.request.Request(
            f"{API_ROOT}/graphql", data=payload, headers=headers, method="POST"
        )

    request = build_request()

    for attempt in range(retry_count):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
                if data.get("errors"):
                    raise RuntimeError(json.dumps(data["errors"], ensure_ascii=False))
                return data["data"], {k.lower(): v for k, v in response.headers.items()}
        except urllib.error.HTTPError as exc:
            remaining = exc.headers.get("X-RateLimit-Remaining")
            reset_at = exc.headers.get("X-RateLimit-Reset")
            if exc.code in {403, 429} and remaining == "0" and reset_at:
                sleep_s = max(int(reset_at) - int(time.time()) + 5, 5)
                print(f"graphql rate limited; sleeping {sleep_s}s", file=sys.stderr)
                time.sleep(sleep_s)
                continue
            if exc.code in {401, 403} and attempt + 1 < retry_count:
                new = refresh_active_token(token)
                if new:
                    token = new
                    request = build_request()
                    continue
            if exc.code >= 500 and attempt + 1 < retry_count:
                time.sleep(2**attempt)
                continue
            raise
        except (http.client.HTTPException, OSError, RuntimeError):
            # See api_get: catch raw connection errors plus GraphQL-error
            # RuntimeErrors so transient blips back off and retry.
            if attempt + 1 >= retry_count:
                raise
            time.sleep(min(2**attempt, 60))
    raise RuntimeError("failed to fetch GitHub GraphQL data")


def next_link(headers: dict[str, str]) -> str | None:
    link_header = headers.get("link", "")
    for part in link_header.split(","):
        if 'rel="next"' in part:
            match = re.search(r"<([^>]+)>", part)
            if match:
                return match.group(1)
    return None


def code_language_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in EXTENSION_LANGUAGE:
        return EXTENSION_LANGUAGE[suffix]
    if "/" in path and "." not in Path(path).name:
        return "extensionless"
    return suffix.lstrip(".") or "unknown"


def natural_language_hint(text: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh_or_cjk"
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"
    if re.search(r"[^\x00-\x7f]", text):
        return "non_ascii_other"
    return "en_or_ascii"


def categories_for_thread(thread: dict[str, Any]) -> list[str]:
    haystack_parts = [
        thread.get("path", ""),
        thread.get("diff_hunk", ""),
        thread.get("pull_request", {}).get("title", ""),
    ]
    haystack_parts.extend(comment.get("body", "") for comment in thread["comments"])
    haystack = "\n".join(haystack_parts).lower()
    labels = []
    for category, patterns in CATEGORY_PATTERNS.items():
        if any(re.search(pattern, haystack) for pattern in patterns):
            labels.append(category)
    return labels or ["general-review"]


def finalize_threads(threads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    finalized = []
    for thread in threads:
        thread["comments"].sort(key=lambda item: item.get("created_at") or "")
        thread["categories"] = categories_for_thread(thread)
        thread["human_reviewer_comment_count"] = sum(
            1 for comment in thread["comments"] if not comment["author"]["is_agent"]
        )
        thread["agent_reviewer_comment_count"] = sum(
            1 for comment in thread["comments"] if comment["author"]["is_agent"]
        )
        thread["comment_language_hints"] = sorted(
            {comment["body_language_hint"] for comment in thread["comments"]}
        )
        if thread["comments"]:
            finalized.append(thread)
    return finalized


def compact_pull(pr: dict[str, Any]) -> dict[str, Any]:
    user = pr.get("user") or {}
    return {
        "number": pr["number"],
        "node_id": pr.get("node_id"),
        "title": pr.get("title") or "",
        "state": pr.get("state"),
        "draft": pr.get("draft"),
        "created_at": pr.get("created_at"),
        "updated_at": pr.get("updated_at"),
        "closed_at": pr.get("closed_at"),
        "merged_at": pr.get("merged_at"),
        "html_url": pr.get("html_url"),
        "labels": [
            label.get("name")
            for label in pr.get("labels", [])
            if isinstance(label, dict) and label.get("name")
        ],
        "author": {
            "login": user.get("login"),
            "type": user.get("type"),
            "is_agent": is_agent_user(user),
        },
    }


def compact_comment(comment: dict[str, Any], kind: str) -> dict[str, Any]:
    user = comment.get("user") or {}
    body = comment.get("body") or ""
    return {
        "kind": kind,
        "id": comment.get("id"),
        "node_id": comment.get("node_id"),
        "in_reply_to_id": comment.get("in_reply_to_id"),
        "created_at": comment.get("created_at"),
        "updated_at": comment.get("updated_at"),
        "html_url": comment.get("html_url"),
        "author": {
            "login": user.get("login"),
            "type": user.get("type"),
            "is_agent": is_agent_user(user),
            "association": comment.get("author_association"),
        },
        "body": body,
        "body_language_hint": natural_language_hint(body),
    }


def compact_review_submission(review: dict[str, Any]) -> dict[str, Any]:
    author = review.get("author") or {}
    login = author.get("login")
    author_type = author.get("__typename") or author.get("type")
    body = review.get("body") or ""
    created_at = review.get("submittedAt") or review.get("createdAt")
    return {
        "kind": "review_submission",
        "id": review.get("databaseId"),
        "node_id": review.get("id"),
        "state": review.get("state"),
        "commit_id": (review.get("commit") or {}).get("oid"),
        "created_at": created_at,
        "updated_at": review.get("updatedAt"),
        "html_url": review.get("url"),
        "author": {
            "login": login,
            "type": author_type,
            "is_agent": is_agent_user({"login": login, "type": author_type}),
            "association": review.get("authorAssociation"),
        },
        "body": body,
        "body_language_hint": natural_language_hint(body),
    }


def compact_graphql_conversation_comment(comment: dict[str, Any]) -> dict[str, Any]:
    author = comment.get("author") or {}
    login = author.get("login")
    author_type = author.get("__typename") or author.get("type")
    body = comment.get("body") or ""
    return {
        "kind": "pr_conversation",
        "id": comment.get("databaseId"),
        "node_id": comment.get("id"),
        "created_at": comment.get("createdAt"),
        "updated_at": comment.get("updatedAt"),
        "html_url": comment.get("url"),
        "author": {
            "login": login,
            "type": author_type,
            "is_agent": is_agent_user({"login": login, "type": author_type}),
            "association": comment.get("authorAssociation"),
        },
        "body": body,
        "body_language_hint": natural_language_hint(body),
    }


def pr_number_from_comment(comment: dict[str, Any]) -> int:
    return int(str(comment["pull_request_url"]).rstrip("/").split("/")[-1])


def issue_number_from_comment(comment: dict[str, Any]) -> int:
    return int(str(comment["issue_url"]).rstrip("/").split("/")[-1])


def fetch_target_pulls(
    repo: str,
    token: str | None,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    max_pages: int | None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    encoded = urllib.parse.quote(repo, safe="/")
    url = (
        f"{API_ROOT}/repos/{encoded}/pulls"
        "?state=all&sort=created&direction=asc&per_page=100"
    )
    target_pulls: dict[int, dict[str, Any]] = {}
    stats = collections.Counter()
    page = 0

    while url:
        page += 1
        if max_pages is not None and page > max_pages:
            break
        pulls, headers = api_get(url, token)
        if not pulls:
            break

        page_created = [parse_github_dt(pr["created_at"]) for pr in pulls]
        if page % 25 == 0:
            print(
                f"pulled PR page {page}: {pulls[0]['created_at']}..{pulls[-1]['created_at']}",
                file=sys.stderr,
            )

        for pr, created_at in zip(pulls, page_created):
            if created_at < start_dt:
                continue
            if created_at > end_dt:
                stats["seen_after_window"] += 1
                continue

            stats["window_prs"] += 1
            compact = compact_pull(pr)
            if compact["author"]["is_agent"]:
                stats["excluded_agent_prs"] += 1
                continue
            stats["included_human_prs"] += 1
            stats[f"included_human_prs_{created_at.year}"] += 1
            target_pulls[pr["number"]] = compact

        if page_created and page_created[0] > end_dt:
            break
        url = next_link(headers)

    return target_pulls, dict(stats)


def fetch_review_threads(
    repo: str,
    token: str | None,
    target_pulls: dict[int, dict[str, Any]],
    include_agent_reviewers: bool,
    end_dt: dt.datetime,
    max_pages: int | None,
    window_label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    encoded = urllib.parse.quote(repo, safe="/")
    url = (
        f"{API_ROOT}/repos/{encoded}/pulls/comments"
        "?sort=created&direction=asc&per_page=100"
    )
    by_thread: dict[int, dict[str, Any]] = {}
    stats = collections.Counter()
    page = 0

    while url:
        page += 1
        if max_pages is not None and page > max_pages:
            break
        comments, headers = api_get(url, token)
        if not comments:
            break
        if page % 25 == 0:
            print(
                "pulled review-comment page "
                f"{page}: {comments[0]['created_at']}..{comments[-1]['created_at']}",
                file=sys.stderr,
            )

        stop_after_page = False
        for comment in comments:
            created_at = parse_github_dt(comment["created_at"])
            if created_at > end_dt:
                stop_after_page = True
                break
            stats["all_review_comments_seen"] += 1
            pr_number = pr_number_from_comment(comment)
            pull = target_pulls.get(pr_number)
            if pull is None:
                continue

            user_is_agent = is_agent_user(comment.get("user"))
            if user_is_agent:
                stats["agent_reviewer_comments_on_target_prs"] += 1
                if not include_agent_reviewers:
                    continue
            else:
                stats["human_reviewer_comments_on_target_prs"] += 1

            root_id = int(comment.get("in_reply_to_id") or comment["id"])
            thread = by_thread.get(root_id)
            if thread is None:
                thread = {
                    "schema_version": 2,
                    "row_type": "inline_review_thread",
                    "source": {
                        "repo": repo,
                        "years": window_label,
                        "collection": "github_pull_review_comments",
                    },
                    "thread_id": root_id,
                    "pull_request": pull,
                    "path": comment.get("path") or "",
                    "code_language": code_language_for_path(comment.get("path") or ""),
                    "diff_hunk": comment.get("diff_hunk") or "",
                    "commit_id": comment.get("commit_id"),
                    "original_commit_id": comment.get("original_commit_id"),
                    "line": comment.get("line"),
                    "original_line": comment.get("original_line"),
                    "start_line": comment.get("start_line"),
                    "original_start_line": comment.get("original_start_line"),
                    "side": comment.get("side"),
                    "subject_type": comment.get("subject_type"),
                    "comments": [],
                }
                by_thread[root_id] = thread
            elif not thread.get("diff_hunk") and comment.get("diff_hunk"):
                thread["diff_hunk"] = comment.get("diff_hunk") or ""
                thread["path"] = comment.get("path") or thread["path"]
                thread["code_language"] = code_language_for_path(thread["path"])

            thread["comments"].append(compact_comment(comment, "inline_review_comment"))

        url = None if stop_after_page else next_link(headers)

    threads = finalize_threads(list(by_thread.values()))

    threads.sort(
        key=lambda item: (
            item["pull_request"]["number"],
            item["comments"][0]["created_at"] if item["comments"] else "",
            item["thread_id"],
        )
    )
    stats["threads"] = len(threads)
    return threads, dict(stats)


PR_CONVERSATIONS_QUERY = """
query($ids: [ID!]!) {
  nodes(ids: $ids) {
    ... on PullRequest {
      number
      comments(first: 100) {
        totalCount
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          databaseId
          id
          body
          createdAt
          updatedAt
          url
          authorAssociation
          author {
            login
            __typename
          }
        }
      }
    }
  }
}
"""


PR_CONVERSATIONS_PAGE_QUERY = """
query($id: ID!, $after: String) {
  node(id: $id) {
    ... on PullRequest {
      number
      comments(first: 100, after: $after) {
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          databaseId
          id
          body
          createdAt
          updatedAt
          url
          authorAssociation
          author {
            login
            __typename
          }
        }
      }
    }
  }
}
"""


def fetch_remaining_connection_nodes(
    node_id: str,
    after_cursor: str | None,
    query: str,
    connection: str,
    token: str | None,
) -> list[dict[str, Any]]:
    """Page through a PullRequest sub-connection beyond the first GraphQL page."""
    collected: list[dict[str, Any]] = []
    cursor = after_cursor
    while cursor:
        data, _headers = api_graphql(query, {"id": node_id, "after": cursor}, token)
        node = data.get("node") or {}
        conn = node.get(connection) or {}
        collected.extend(conn.get("nodes") or [])
        page_info = conn.get("pageInfo") or {}
        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
    return collected


def fetch_pr_conversations(
    repo: str,
    token: str | None,
    target_pulls: dict[int, dict[str, Any]],
    include_agent_reviewers: bool,
    end_dt: dt.datetime,
    batch_size: int,
    max_batches: int | None,
    window_label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("--graphql-batch-size must be positive")

    pulls = [pull for pull in target_pulls.values() if pull.get("node_id")]
    by_number = {pull["number"]: pull for pull in pulls}
    by_pr: dict[int, dict[str, Any]] = {}
    stats = collections.Counter()

    def ingest(comment: dict[str, Any], pr_number: int, pull: dict[str, Any]) -> None:
        compact = compact_graphql_conversation_comment(comment)
        created_at = compact.get("created_at")
        if created_at and parse_github_dt(created_at) > end_dt:
            stats["pr_conversation_comments_after_window_skipped"] += 1
            return

        if compact["author"]["is_agent"]:
            stats["agent_pr_conversation_comments_on_target_prs"] += 1
            if not include_agent_reviewers:
                return
        else:
            stats["human_pr_conversation_comments_on_target_prs"] += 1

        thread = by_pr.get(pr_number)
        if thread is None:
            thread = {
                "schema_version": 2,
                "row_type": "pr_conversation",
                "source": {
                    "repo": repo,
                    "years": window_label,
                    "collection": "github_pr_issue_comments",
                },
                "thread_id": f"pr-conversation-{pr_number}",
                "pull_request": pull,
                "path": "",
                "code_language": "conversation",
                "diff_hunk": "",
                "commit_id": None,
                "original_commit_id": None,
                "line": None,
                "original_line": None,
                "start_line": None,
                "original_start_line": None,
                "side": None,
                "subject_type": "pull_request",
                "comments": [],
            }
            by_pr[pr_number] = thread

        thread["comments"].append(compact)

    for batch_index, start in enumerate(range(0, len(pulls), batch_size), start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        batch = pulls[start : start + batch_size]
        if batch_index % 25 == 0:
            print(
                f"pulled conversation batch {batch_index}: "
                f"PR {batch[0]['number']}..{batch[-1]['number']}",
                file=sys.stderr,
            )

        data, _headers = api_graphql(
            PR_CONVERSATIONS_QUERY,
            {"ids": [pull["node_id"] for pull in batch]},
            token,
        )
        for node in data.get("nodes", []):
            if not node:
                continue
            pr_number = int(node["number"])
            pull = by_number[pr_number]
            comments = node.get("comments") or {}
            stats["all_pr_conversation_comments_seen"] += (
                comments.get("totalCount") or 0
            )

            for comment in comments.get("nodes") or []:
                ingest(comment, pr_number, pull)

            page_info = comments.get("pageInfo") or {}
            if page_info.get("hasNextPage"):
                stats["pr_conversation_paginated_prs"] += 1
                for comment in fetch_remaining_connection_nodes(
                    pull["node_id"],
                    page_info.get("endCursor"),
                    PR_CONVERSATIONS_PAGE_QUERY,
                    "comments",
                    token,
                ):
                    ingest(comment, pr_number, pull)

    threads = finalize_threads(list(by_pr.values()))
    threads.sort(
        key=lambda item: (
            item["pull_request"]["number"],
            item["comments"][0]["created_at"] if item["comments"] else "",
            item["thread_id"],
        )
    )
    stats["threads"] = len(threads)
    return threads, dict(stats)


REVIEW_SUBMISSIONS_QUERY = """
query($ids: [ID!]!) {
  nodes(ids: $ids) {
    ... on PullRequest {
      number
      reviews(first: 100) {
        totalCount
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          databaseId
          id
          state
          body
          submittedAt
          createdAt
          updatedAt
          url
          authorAssociation
          commit {
            oid
          }
          author {
            login
            __typename
          }
        }
      }
    }
  }
}
"""


REVIEW_SUBMISSIONS_PAGE_QUERY = """
query($id: ID!, $after: String) {
  node(id: $id) {
    ... on PullRequest {
      number
      reviews(first: 100, after: $after) {
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          databaseId
          id
          state
          body
          submittedAt
          createdAt
          updatedAt
          url
          authorAssociation
          commit {
            oid
          }
          author {
            login
            __typename
          }
        }
      }
    }
  }
}
"""


def fetch_review_submissions(
    repo: str,
    token: str | None,
    target_pulls: dict[int, dict[str, Any]],
    include_agent_reviewers: bool,
    end_dt: dt.datetime,
    batch_size: int,
    max_batches: int | None,
    window_label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("--graphql-batch-size must be positive")

    pulls = [pull for pull in target_pulls.values() if pull.get("node_id")]
    by_number = {pull["number"]: pull for pull in pulls}
    by_pr: dict[int, dict[str, Any]] = {}
    stats = collections.Counter()

    def ingest(review: dict[str, Any], pr_number: int, pull: dict[str, Any]) -> None:
        compact = compact_review_submission(review)
        created_at = compact.get("created_at")
        if created_at and parse_github_dt(created_at) > end_dt:
            stats["review_submissions_after_window_skipped"] += 1
            return
        body = compact.get("body") or ""
        if not body.strip() and compact.get("state") not in {
            "REQUEST_CHANGES",
            "CHANGES_REQUESTED",
        }:
            stats["empty_review_submissions_skipped"] += 1
            return
        if compact["author"]["is_agent"]:
            stats["agent_review_submissions_on_target_prs"] += 1
            if not include_agent_reviewers:
                return
        else:
            stats["human_review_submissions_on_target_prs"] += 1

        thread = by_pr.get(pr_number)
        if thread is None:
            thread = {
                "schema_version": 2,
                "row_type": "review_submission",
                "source": {
                    "repo": repo,
                    "years": window_label,
                    "collection": "github_pull_request_reviews",
                },
                "thread_id": f"review-submissions-{pr_number}",
                "pull_request": pull,
                "path": "",
                "code_language": "review",
                "diff_hunk": "",
                "commit_id": None,
                "original_commit_id": None,
                "line": None,
                "original_line": None,
                "start_line": None,
                "original_start_line": None,
                "side": None,
                "subject_type": "pull_request_review",
                "comments": [],
            }
            by_pr[pr_number] = thread

        thread["comments"].append(compact)

    for batch_index, start in enumerate(range(0, len(pulls), batch_size), start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        batch = pulls[start : start + batch_size]
        if batch_index % 25 == 0:
            print(
                f"pulled review-submission batch {batch_index}: "
                f"PR {batch[0]['number']}..{batch[-1]['number']}",
                file=sys.stderr,
            )

        data, _headers = api_graphql(
            REVIEW_SUBMISSIONS_QUERY,
            {"ids": [pull["node_id"] for pull in batch]},
            token,
        )
        for node in data.get("nodes", []):
            if not node:
                continue
            pr_number = int(node["number"])
            pull = by_number[pr_number]
            reviews = node.get("reviews") or {}
            stats["all_review_submissions_seen"] += reviews.get("totalCount") or 0

            for review in reviews.get("nodes") or []:
                ingest(review, pr_number, pull)

            page_info = reviews.get("pageInfo") or {}
            if page_info.get("hasNextPage"):
                stats["review_submission_paginated_prs"] += 1
                for review in fetch_remaining_connection_nodes(
                    pull["node_id"],
                    page_info.get("endCursor"),
                    REVIEW_SUBMISSIONS_PAGE_QUERY,
                    "reviews",
                    token,
                ):
                    ingest(review, pr_number, pull)

    threads = finalize_threads(list(by_pr.values()))
    threads.sort(
        key=lambda item: (
            item["pull_request"]["number"],
            item["comments"][0]["created_at"] if item["comments"] else "",
            item["thread_id"],
        )
    )
    stats["threads"] = len(threads)
    return threads, dict(stats)


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def prefixed_stats(prefix: str, stats: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def summarize(
    threads: list[dict[str, Any]],
    pull_stats: dict[str, Any],
    comment_stats: dict[str, Any],
    repo: str,
    start_year: int,
    end_year: int,
    include_agent_reviewers: bool,
) -> dict[str, Any]:
    category_counts: collections.Counter[str] = collections.Counter()
    path_counts: collections.Counter[str] = collections.Counter()
    language_counts: collections.Counter[str] = collections.Counter()
    body_language_counts: collections.Counter[str] = collections.Counter()
    reviewer_counts: collections.Counter[str] = collections.Counter()
    pr_year_counts: collections.Counter[str] = collections.Counter()
    row_type_counts: collections.Counter[str] = collections.Counter()
    event_kind_counts: collections.Counter[str] = collections.Counter()
    review_state_counts: collections.Counter[str] = collections.Counter()
    comments_per_thread: list[int] = []

    for thread in threads:
        row_type_counts[thread.get("row_type", "inline_review_thread")] += 1
        for category in thread["categories"]:
            category_counts[category] += 1
        path_counts[thread["path"] or "<conversation>"] += 1
        language_counts[thread["code_language"]] += 1
        pr_year = thread["pull_request"]["created_at"][:4]
        pr_year_counts[pr_year] += 1
        comments_per_thread.append(len(thread["comments"]))
        for comment in thread["comments"]:
            event_kind_counts[comment.get("kind", "inline_review_comment")] += 1
            if comment.get("state"):
                review_state_counts[comment["state"]] += 1
            body_language_counts[comment["body_language_hint"]] += 1
            if not comment["author"]["is_agent"]:
                reviewer_counts[comment["author"]["login"] or "unknown"] += 1

    return {
        "schema_version": 2,
        "repo": repo,
        "generated_at": iso_utc_now(),
        "source_years": [start_year, end_year],
        "collection_policy": {
            "pull_requests": "PR created_at within inclusive year range; bot and coding-agent PR authors excluded.",
            "event_window": "Inline comments, PR conversation comments, and review submissions are also capped at end-year 12-31 23:59:59 UTC.",
            "inline_review_comments": "GitHub pull review comments grouped by in_reply_to_id thread.",
            "pr_conversation_comments": "Top-level PR conversation comments from GitHub PullRequest comments, grouped by PR.",
            "review_submissions": "Pull request review submission bodies, grouped by PR. Empty approvals are skipped unless the review requested changes.",
            "agent_reviewers_included": include_agent_reviewers,
            "comment_text_policy": "Original comment bodies are preserved as UTF-8; no natural-language filter is applied.",
        },
        "pull_request_stats": pull_stats,
        "comment_stats": comment_stats,
        "thread_count": len(threads),
        "comment_count": sum(len(thread["comments"]) for thread in threads),
        "human_reviewer_comment_count": sum(
            thread["human_reviewer_comment_count"] for thread in threads
        ),
        "agent_reviewer_comment_count": sum(
            thread["agent_reviewer_comment_count"] for thread in threads
        ),
        "threads_by_pr_year": dict(sorted(pr_year_counts.items())),
        "threads_by_type": row_type_counts.most_common(),
        "events_by_kind": event_kind_counts.most_common(),
        "review_states": review_state_counts.most_common(),
        "top_categories": category_counts.most_common(20),
        "top_paths": path_counts.most_common(30),
        "code_languages": language_counts.most_common(),
        "comment_language_hints": body_language_counts.most_common(),
        "top_human_reviewers": reviewer_counts.most_common(30),
        "max_comments_per_thread": max(comments_per_thread, default=0),
    }


def write_summary_markdown(
    path: Path, metadata: dict[str, Any], corpus_name: str
) -> None:
    def table(rows: list[tuple[Any, Any]], left: str, right: str) -> list[str]:
        lines = [f"| {left} | {right} |", "| --- | ---: |"]
        lines.extend(f"| `{name}` | {count} |" for name, count in rows)
        return lines

    lines = [
        "# SGLang Human Review Corpus Summary",
        "",
        f"- Repo: `{metadata['repo']}`",
        f"- Source PR years: `{metadata['source_years'][0]}` to `{metadata['source_years'][1]}` inclusive",
        f"- Generated at: `{metadata['generated_at']}`",
        f"- Corpus file: `{corpus_name}`",
        f"- Threads: `{metadata['thread_count']}`",
        f"- Comments in corpus: `{metadata['comment_count']}`",
        f"- Human reviewer comments: `{metadata['human_reviewer_comment_count']}`",
        f"- Agent reviewer comments: `{metadata['agent_reviewer_comment_count']}`",
        "",
        "## Collection Policy",
        "",
        "- Pull requests are selected by PR `created_at` in the requested year range.",
        "- Review/comment events are also capped at the requested end date.",
        "- Pull requests authored by GitHub bots or obvious coding-agent accounts are excluded.",
        "- Inline review comments are GitHub pull-review comments grouped by thread.",
        "- Top-level PR conversation comments are included from GitHub PR comments and grouped by PR.",
        "- Review submission bodies are included for COMMENT / REQUEST_CHANGES style review summaries; empty approvals are skipped.",
        "- Comment bodies are kept in their original language; the corpus does not translate or drop non-English text.",
        "- `diff_hunk` stores the code context that produced each review thread.",
        "",
        "## Pull Request Stats",
        "",
    ]
    lines.extend(
        table(sorted(metadata["pull_request_stats"].items()), "Metric", "Count")
    )
    lines.extend(["", "## Comment Stats", ""])
    lines.extend(table(sorted(metadata["comment_stats"].items()), "Metric", "Count"))
    lines.extend(["", "## Episode Types", ""])
    lines.extend(table(metadata["threads_by_type"], "Type", "Threads"))
    lines.extend(["", "## Event Kinds", ""])
    lines.extend(table(metadata["events_by_kind"], "Kind", "Events"))
    if metadata["review_states"]:
        lines.extend(["", "## Review States", ""])
        lines.extend(table(metadata["review_states"], "State", "Events"))
    lines.extend(["", "## Top Categories", ""])
    lines.extend(table(metadata["top_categories"], "Category", "Threads"))
    lines.extend(["", "## Code Languages", ""])
    lines.extend(table(metadata["code_languages"], "Language", "Threads"))
    lines.extend(["", "## Comment Language Hints", ""])
    lines.extend(table(metadata["comment_language_hints"], "Hint", "Comments"))
    lines.extend(["", "## Top Paths", ""])
    lines.extend(table(metadata["top_paths"], "Path", "Threads"))
    lines.extend(["", "## Top Human Reviewers", ""])
    lines.extend(table(metadata["top_human_reviewers"], "Reviewer", "Comments"))
    lines.extend(
        [
            "",
            "## Query Examples",
            "",
            "```bash",
            "python3 skills/sglang-humanize-review/scripts/query_sglang_review_corpus.py --query cuda --limit 5",
            "python3 skills/sglang-humanize-review/scripts/query_sglang_review_corpus.py --path python/sglang/srt --category correctness --limit 8",
            "python3 skills/sglang-humanize-review/scripts/query_sglang_review_corpus.py --query 'server_args' --format jsonl --limit 3",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    # Prefer a freshly-read gh token: long runs can outlive a stale env snapshot.
    token = gh_token_from_cli() or args.token
    if not token:
        print("error: provide GITHUB_TOKEN or authenticate gh", file=sys.stderr)
        return 2
    set_active_token(token)

    start_dt = (
        dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
        if args.from_beginning
        else dt.datetime(args.start_year, 1, 1, tzinfo=dt.timezone.utc)
    )
    end_dt = dt.datetime(args.end_year, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc)
    # Never collect events from the future; cap the window at "now".
    now_dt = dt.datetime.now(dt.timezone.utc)
    if end_dt > now_dt:
        end_dt = now_dt
    window_label = (
        f"{start_dt.year if not args.from_beginning else 'start'}-{end_dt.year}"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"collecting PRs from {args.repo} created {start_dt.isoformat()}..{end_dt.isoformat()}",
        file=sys.stderr,
    )
    pulls, pull_stats = fetch_target_pulls(
        args.repo,
        token,
        start_dt,
        end_dt,
        args.max_pr_pages,
    )
    print(f"target human PRs: {len(pulls)}", file=sys.stderr)

    all_threads: list[dict[str, Any]] = []
    combined_comment_stats: dict[str, Any] = {}

    if args.include_inline_review_comments:
        print("collecting global pull review comments", file=sys.stderr)
        inline_threads, inline_stats = fetch_review_threads(
            args.repo,
            token,
            pulls,
            args.include_agent_reviewers,
            end_dt,
            args.max_comment_pages,
            window_label,
        )
        all_threads.extend(inline_threads)
        combined_comment_stats.update(prefixed_stats("inline", inline_stats))

    if args.include_pr_conversations:
        print("collecting top-level PR conversation comments", file=sys.stderr)
        conversation_threads, conversation_stats = fetch_pr_conversations(
            args.repo,
            token,
            pulls,
            args.include_agent_reviewers,
            end_dt,
            args.graphql_batch_size,
            args.max_issue_comment_pages,
            window_label,
        )
        all_threads.extend(conversation_threads)
        combined_comment_stats.update(
            prefixed_stats("conversation", conversation_stats)
        )

    if args.include_review_submissions:
        print("collecting pull request review submissions", file=sys.stderr)
        submission_threads, submission_stats = fetch_review_submissions(
            args.repo,
            token,
            pulls,
            args.include_agent_reviewers,
            end_dt,
            args.graphql_batch_size,
            args.max_review_submission_batches,
            window_label,
        )
        all_threads.extend(submission_threads)
        combined_comment_stats.update(prefixed_stats("submission", submission_stats))

    all_threads.sort(
        key=lambda item: (
            item["pull_request"]["number"],
            item["comments"][0]["created_at"] if item.get("comments") else "",
            str(item["thread_id"]),
        )
    )

    corpus_path = args.out_dir / args.out_name
    metadata_path = args.out_dir / args.metadata_name
    summary_path = args.out_dir / "corpus-summary.md"

    summary_start_year = (
        min((t["pull_request"]["created_at"][:4] for t in all_threads), default=None)
        if args.from_beginning
        else args.start_year
    )
    summary_start_year = (
        int(summary_start_year) if summary_start_year else args.start_year
    )

    write_jsonl_gz(corpus_path, all_threads)
    metadata = summarize(
        all_threads,
        pull_stats,
        combined_comment_stats,
        args.repo,
        summary_start_year,
        end_dt.year,
        args.include_agent_reviewers,
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary_markdown(summary_path, metadata, corpus_path.name)

    print(f"wrote {corpus_path}", file=sys.stderr)
    print(f"wrote {metadata_path}", file=sys.stderr)
    print(f"wrote {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
