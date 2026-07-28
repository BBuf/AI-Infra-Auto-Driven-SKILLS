#!/usr/bin/env python3
"""Collect mechanical metadata for allowlisted public GitHub pull requests."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_ALLOWED_REPOSITORIES = {"sgl-project/sglang"}
PR_URL_PATTERN = re.compile(
    r"https://github\.com/([A-Za-z0-9_.-]+)/" r"([A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)/?"
)


@dataclasses.dataclass(frozen=True)
class PublicPR:
    repository: str
    number: int
    url: str


def parse_pr_url(url: str, allowed_repositories: set[str]) -> PublicPR:
    """Parse a canonical HTTPS PR URL and enforce a repository allowlist."""
    match = PR_URL_PATTERN.fullmatch(url)
    if match is None:
        raise ValueError(f"not a canonical public GitHub PR URL: {url}")
    repository = f"{match.group(1)}/{match.group(2)}"
    if repository not in allowed_repositories:
        raise ValueError(f"repository is not allowed: {repository}")
    return PublicPR(repository, int(match.group(3)), url.rstrip("/"))


def _state(payload: dict[str, Any]) -> str:
    if payload.get("merged_at"):
        return "merged"
    if payload.get("state") == "open":
        return "open"
    return "closed"


def _file_record(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "filename": payload["filename"],
        "status": payload.get("status"),
        "additions": payload.get("additions", 0),
        "deletions": payload.get("deletions", 0),
        "changes": payload.get("changes", 0),
        "previous_filename": payload.get("previous_filename"),
    }


def build_record(
    pr_payload: dict[str, Any],
    file_payloads: Iterable[dict[str, Any]],
    *,
    captured_at: str,
) -> dict[str, Any]:
    """Create a mechanical record without interpreting the PR."""
    files = sorted(
        (_file_record(payload) for payload in file_payloads),
        key=lambda item: item["filename"],
    )
    return {
        "schema_version": 1,
        "captured_at": captured_at,
        "repository": "/".join(pr_payload["html_url"].split("/")[3:5]),
        "number": pr_payload["number"],
        "url": pr_payload["html_url"],
        "title": pr_payload["title"],
        "state": _state(pr_payload),
        "draft": bool(pr_payload.get("draft")),
        "created_at": pr_payload.get("created_at"),
        "updated_at": pr_payload.get("updated_at"),
        "closed_at": pr_payload.get("closed_at"),
        "merged_at": pr_payload.get("merged_at"),
        "head_sha": pr_payload["head"]["sha"],
        "base_sha": pr_payload["base"]["sha"],
        "additions": pr_payload.get("additions", 0),
        "deletions": pr_payload.get("deletions", 0),
        "changed_files": pr_payload.get("changed_files", len(files)),
        "files": files,
    }


def _gh_api(endpoint: str) -> Any:
    completed = subprocess.run(
        ["gh", "api", endpoint],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def fetch_record(pr: PublicPR, *, captured_at: str) -> dict[str, Any]:
    """Fetch PR metadata and its complete paginated file inventory."""
    prefix = f"repos/{pr.repository}/pulls/{pr.number}"
    pr_payload = _gh_api(prefix)
    file_payloads: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = _gh_api(f"{prefix}/files?per_page=100&page={page}")
        if not isinstance(payload, list):
            raise ValueError(f"unexpected files response for {pr.url}")
        file_payloads.extend(payload)
        if len(payload) < 100:
            break
        page += 1
    return build_record(pr_payload, file_payloads, captured_at=captured_at)


def _utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def collect(
    urls: Sequence[str],
    *,
    allowed_repositories: set[str],
    captured_at: str,
) -> dict[str, Any]:
    parsed = [parse_pr_url(url, allowed_repositories) for url in urls]
    records = [fetch_record(pr, captured_at=captured_at) for pr in parsed]
    records.sort(key=lambda item: (item["repository"], item["number"]))
    return {
        "schema_version": 1,
        "captured_at": captured_at,
        "allowed_repositories": sorted(allowed_repositories),
        "pull_requests": records,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect mechanical public GitHub PR metadata. "
            "Manually review each diff before drawing technical conclusions."
        )
    )
    parser.add_argument("urls", nargs="+", help="Canonical public GitHub PR URLs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-repository",
        action="append",
        dest="allowed_repositories",
        help="Allow an owner/repository value; defaults to sgl-project/sglang",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    allowed = set(args.allowed_repositories or DEFAULT_ALLOWED_REPOSITORIES)
    captured_at = _utc_now()
    payload = collect(
        args.urls,
        allowed_repositories=allowed,
        captured_at=captured_at,
    )
    _write_json_atomic(args.output, payload)
    print(
        f"Wrote {len(payload['pull_requests'])} public PR record(s) "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
