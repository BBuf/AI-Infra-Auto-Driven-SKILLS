#!/usr/bin/env python3
"""Resolve public original model architecture diagrams from the skill index."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

SKILL_DIR = Path(__file__).resolve().parents[1]
INDEX_PATH = SKILL_DIR / "references" / "diagram-index.json"
LOCAL_SOURCE_ROOTS = {
    "InfraTech": Path("/tmp/InfraTech"),
    "self-llm": Path("/tmp/self-llm"),
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("_", "-").replace("/", " ")
    text = re.sub(r"[^a-z0-9.\-\u4e00-\u9fff]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", normalize(text))


def load_index() -> dict[str, Any]:
    with INDEX_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_contains(needle: str, haystack: str) -> bool:
    """Substring match that avoids version-prefix false positives.

    `deepseek-v3` should not match `deepseek-v3.2`, while `deepseek mla`
    should match `deepseek mla mha`.
    """

    pos = haystack.find(needle)
    if pos < 0:
        return False
    before = haystack[pos - 1] if pos > 0 else " "
    after_pos = pos + len(needle)
    after = haystack[after_pos] if after_pos < len(haystack) else " "
    if before.isalnum():
        return False
    if after.isalnum():
        return False
    if after in ".- " and after_pos + 1 < len(haystack) and needle[-1].isdigit():
        suffix = haystack[after_pos + 1 :]
        suffix_token = suffix.replace("-", " ").split(" ", 1)[0]
        if suffix_token not in {
            "architecture",
            "diagram",
            "model",
            "fp8",
            "bf16",
            "nvfp4",
            "awq",
            "gptq",
            "instruct",
            "thinking",
            "pt",
            "chat",
            "base",
        }:
            return False
    return True


def alias_score(query: str, aliases: list[str], title: str) -> int:
    q = normalize(query)
    qc = compact(query)
    candidates = [normalize(title), *(normalize(alias) for alias in aliases)]
    best = 0
    for candidate in candidates:
        if not candidate:
            continue
        cc = compact(candidate)
        if q == candidate:
            best = max(best, 100)
        elif qc and qc == cc:
            best = max(best, 95)
        elif safe_contains(q, candidate) or safe_contains(candidate, q):
            best = max(best, 80)
        else:
            q_tokens = set(q.replace("-", " ").split())
            c_tokens = set(candidate.replace("-", " ").split())
            if q_tokens and c_tokens:
                overlap = len(q_tokens & c_tokens)
                if overlap:
                    best = max(best, 20 + overlap * 10)
    return best


def find_existing(query: str, max_results: int) -> list[dict[str, Any]]:
    index = load_index()
    matches: list[tuple[int, int, dict[str, Any]]] = []
    for entry in index["diagrams"]:
        score = alias_score(query, entry.get("aliases", []), entry["title"])
        if score >= 70:
            rank = int(entry.get("rank", 100))
            matches.append((score, -rank, entry))
    matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [entry for _, _, entry in matches[:max_results]]


def result_existing(query: str, matches: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "kind": "existing",
        "query": query,
        "diagrams": [
            {
                "title": match["title"],
                "url": match["url"],
                "source": match["source"],
                "path": match["path"],
                "local_path": (
                    str(LOCAL_SOURCE_ROOTS[match["source"]] / match["path"])
                    if match["source"] in LOCAL_SOURCE_ROOTS
                    else None
                ),
            }
            for match in matches
        ],
    }


def result_no_match(query: str) -> dict[str, Any]:
    return {
        "kind": "no_match",
        "query": query,
        "message": "No public original architecture diagram is indexed for this model.",
    }


def markdown(result: dict[str, Any]) -> str:
    if result["kind"] == "existing":
        lines = [f"## {result['query']} architecture diagrams", ""]
        for item in result["diagrams"]:
            lines.append(f"![{item['title']}]({item['url']})")
            lines.append("")
            lines.append(f"Source: {item['source']} `{item['path']}`")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    return f"## {result['query']} architecture diagram\n\n" f"{result['message']}\n"


def list_known(index: dict[str, Any]) -> str:
    lines = []
    for entry in sorted(index["diagrams"], key=lambda e: (e["source"], e["title"])):
        aliases = ", ".join(entry.get("aliases", [])[:5])
        lines.append(f"- {entry['title']} [{entry['source']}]: {aliases}")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", help="Model name or Hugging Face model id")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--max-existing", type=int, default=4)
    parser.add_argument("--list-known", action="store_true")
    args = parser.parse_args(argv)

    index = load_index()
    if args.list_known:
        sys.stdout.write(list_known(index))
        return 0

    if not args.model:
        parser.error("model is required unless --list-known is used")

    matches = find_existing(args.model, args.max_existing)
    result = (
        result_existing(args.model, matches) if matches else result_no_match(args.model)
    )

    if args.format == "json":
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
