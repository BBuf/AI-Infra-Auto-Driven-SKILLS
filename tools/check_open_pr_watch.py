#!/usr/bin/env python3
"""Search open upstream PRs that may affect skill guidance.

The output is intentionally a watch list, not a generated truth table. It helps
refresh skill docs before long benchmark, profiler, or model-history work by
surfacing open PRs for the model families and kernel/runtime terms this repo
tracks most closely.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib import error, parse, request


DEFAULT_REPOS = {
    "sglang": "sgl-project/sglang",
    "vllm": "vllm-project/vllm",
    "tensorrt_llm": "NVIDIA/TensorRT-LLM",
    "tokenspeed": "lightseekorg/tokenspeed",
}

DEFAULT_TERMS = [
    "Qwen3.5",
    "Qwen3.6",
    "DeepSeek V4",
    "Kimi K2.5",
    "KimiLinear",
    "GLM-5",
    "GLM-5.2",
    "MLA",
    "GDN",
    "KDA",
    "MoE",
    "FP4",
    "NVFP4",
    "WideEP",
    "DFlash",
]


@dataclass(frozen=True)
class WatchItem:
    repo: str
    number: int
    title: str
    url: str
    updated_at: str
    terms: tuple[str, ...]


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{proc.stderr}")
    return proc.stdout


def search_query(repo: str, term: str) -> str:
    return f"repo:{repo} is:pr is:open {term}"


def combined_search_query(repo: str, terms: list[str]) -> str:
    rendered_terms = [f'"{term}"' if " " in term else term for term in terms]
    return f"repo:{repo} is:pr is:open (" + " OR ".join(rendered_terms) + ")"


def gh_search_open_prs(repo: str, _terms: list[str], per_page: int) -> list[dict[str, Any]]:
    payload = run(
        [
            "gh",
            "api",
            f"/repos/{repo}/pulls?state=open&sort=updated&direction=desc&per_page={min(per_page, 100)}",
        ]
    )
    return json.loads(payload)


def http_fetch_open_prs(repo: str, per_page: int) -> list[dict[str, Any]]:
    query = parse.urlencode(
        {
            "state": "open",
            "sort": "updated",
            "direction": "desc",
            "per_page": str(min(per_page, 100)),
        }
    )
    url = f"https://api.github.com/repos/{repo}/pulls?{query}"
    req = request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "ai-infra-auto-driven-skills-open-pr-watch",
        },
    )
    try:
        with request.urlopen(req, timeout=30.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (OSError, error.HTTPError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"anonymous REST fallback failed for {repo}: {exc}") from exc


def matched_terms_for_item(item: dict[str, Any], terms: list[str]) -> tuple[str, ...]:
    haystack = " ".join(str(item.get(key) or "") for key in ("title", "body")).lower()
    matched = [term for term in terms if term.lower() in haystack]
    return tuple(sorted(matched))


def collect_watch_items(
    repos: dict[str, str],
    terms: list[str],
    per_term_limit: int = 50,
) -> list[WatchItem]:
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    matched_terms: dict[tuple[str, int], set[str]] = {}
    failures: list[str] = []
    succeeded = 0
    for repo in repos.values():
        try:
            repo_items = gh_search_open_prs(repo, terms, per_term_limit)
        except RuntimeError as exc:
            try:
                repo_items = http_fetch_open_prs(repo, per_term_limit)
            except RuntimeError as fallback_exc:
                message = f"{repo}: gh failed ({exc}); fallback failed ({fallback_exc})"
                failures.append(message)
                print(f"warning: skipped {message}", file=sys.stderr)
                continue
            print(f"warning: used anonymous REST fallback for {repo}: {exc}", file=sys.stderr)
        succeeded += 1
        for item in repo_items:
            terms_matched = matched_terms_for_item(item, terms)
            if not terms_matched:
                continue
            number = int(item["number"])
            key = (repo, number)
            by_key[key] = item
            matched_terms.setdefault(key, set()).update(terms_matched)

    watch_items = [
        WatchItem(
            repo=repo,
            number=number,
            title=item.get("title") or "",
            url=item.get("html_url") or f"https://github.com/{repo}/pull/{number}",
            updated_at=(item.get("updated_at") or "")[:10],
            terms=tuple(sorted(matched_terms[(repo, number)])),
        )
        for (repo, number), item in by_key.items()
    ]
    if succeeded == 0 and repos:
        raise RuntimeError("all open PR fetches failed: " + " | ".join(failures))
    return sorted(watch_items, key=lambda item: (item.repo.lower(), item.number))


def render_markdown(items: list[WatchItem], *, generated_on: str) -> str:
    lines = [
        "# Open PR Watch",
        "",
        f"Generated: `{generated_on}`.",
        "",
        "This report is a triage aid for skill updates. Read the linked PR diffs",
        "before changing benchmark, profiler, or model-history guidance.",
        "",
    ]
    repos = sorted({item.repo for item in items})
    if not repos:
        lines.extend(["No open PRs matched the configured watch terms.", ""])
        return "\n".join(lines)

    for repo in repos:
        lines.extend(
            [
                f"## {repo}",
                "",
                "| PR | Updated | Matched terms | Title |",
                "| --- | --- | --- | --- |",
            ]
        )
        for item in [candidate for candidate in items if candidate.repo == repo]:
            terms = ", ".join(f"`{term}`" for term in item.terms)
            title = item.title.replace("|", "\\|")
            lines.append(f"| [#{item.number}]({item.url}) | {item.updated_at} | {terms} | {title} |")
        lines.append("")
    return "\n".join(lines)


def render_json(items: list[WatchItem]) -> str:
    rows = [
        {
            "repo": item.repo,
            "number": item.number,
            "title": item.title,
            "url": item.url,
            "updated_at": item.updated_at,
            "terms": list(item.terms),
        }
        for item in items
    ]
    return json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", action="append", choices=sorted(DEFAULT_REPOS), help="framework repo key to query")
    parser.add_argument("--term", action="append", help="extra search term; defaults remain enabled")
    parser.add_argument(
        "--per-term-limit",
        type=int,
        default=50,
        help="maximum search rows per repository; kept for CLI compatibility",
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repo_keys = args.repo or list(DEFAULT_REPOS)
    repos = {key: DEFAULT_REPOS[key] for key in repo_keys}
    terms = DEFAULT_TERMS + (args.term or [])
    try:
        items = collect_watch_items(repos, terms, args.per_term_limit)
    except RuntimeError as exc:
        parser.exit(2, f"error: {exc}\n")
    text = render_json(items) if args.format == "json" else render_markdown(items, generated_on=date.today().isoformat())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
    else:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")


if __name__ == "__main__":
    main()
