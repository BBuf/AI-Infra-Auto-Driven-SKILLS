#!/usr/bin/env python3
"""Query the model PR optimization history knowledge base."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORKS = ("sglang", "vllm", "tensorrt_llm", "tokenspeed")
LANG_TO_FILES = {
    "en": ("README.en.md",),
    "zh": ("README.zh.md",),
    "both": ("README.en.md", "README.zh.md"),
}


@dataclass(frozen=True)
class Doc:
    framework: str
    model: str
    path: Path
    text: str


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _tokenize(value: str) -> list[str]:
    tokens = [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]
    return list(dict.fromkeys(tokens))


def _preferred_model_slug(query: str) -> str | None:
    normalized = query.lower().replace("_", "-")
    if "qwen3.5" in normalized or "qwen3-5" in normalized or "qwen35" in normalized:
        return "qwen35"
    if "qwen3-next" in normalized:
        return "qwen3-next"
    if "qwen3-coder" in normalized:
        return "qwen3-coder"
    if "qwen3-vl" in normalized or "qwen-vl" in normalized or "qwen3-omni" in normalized:
        return "qwen-vlm-omni-asr"
    if re.search(r"(?:^|[/\-\s])qwen3-\d", normalized) or "qwen/qwen3" in normalized:
        return "qwen3-core"
    return None


def _frameworks(selected: str | None) -> list[str]:
    if selected:
        return [selected]
    return list(FRAMEWORKS)


def _model_dirs(framework: str, selected: str | None) -> list[Path]:
    root = ROOT / framework
    if not root.exists():
        return []
    if selected:
        target = root / _slugify(selected)
        if target.exists():
            return [target]
        tokens = _tokenize(selected)
        return [
            path
            for path in sorted(root.iterdir())
            if path.is_dir() and any(token in path.name.lower() for token in tokens)
        ]
    return [path for path in sorted(root.iterdir()) if path.is_dir()]


def _docs(framework: str | None, model: str | None, lang: str) -> list[Doc]:
    docs: list[Doc] = []
    for fw in _frameworks(framework):
        for model_dir in _model_dirs(fw, model):
            for filename in LANG_TO_FILES[lang]:
                path = model_dir / filename
                if path.exists():
                    docs.append(
                        Doc(
                            framework=fw,
                            model=model_dir.name,
                            path=path,
                            text=path.read_text(encoding="utf-8"),
                        )
                    )
    return docs


def _list_models(framework: str | None) -> None:
    for fw in _frameworks(framework):
        root = ROOT / fw
        if not root.exists():
            continue
        for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            print(f"{fw}/{model_dir.name}")


def _print_paths(docs: list[Doc]) -> None:
    for doc in docs:
        print(doc.path.relative_to(ROOT))


def _score(doc: Doc, terms: list[str], preferred_model: str | None = None) -> int:
    haystack = f"{doc.framework} {doc.model} {doc.path.name}\n{doc.text}".lower()
    model_slug = doc.model.lower()
    score = 0
    if preferred_model and model_slug == preferred_model:
        score += 100000
    for term in terms:
        slug_term = _slugify(term)
        score += haystack.count(term) * 2
        if slug_term and slug_term in model_slug:
            score += 1000
            if any(char.isdigit() for char in slug_term):
                score += 5000
        if term in doc.framework:
            score += 10
    return score


def _matching_lines(doc: Doc, terms: list[str], limit: int = 6) -> list[str]:
    matches: list[str] = []
    for line_no, line in enumerate(doc.text.splitlines(), 1):
        lowered = line.lower()
        if any(term in lowered for term in terms):
            compact = re.sub(r"\s+", " ", line).strip()
            if compact:
                matches.append(f"{doc.path.relative_to(ROOT)}:{line_no}: {compact[:220]}")
        if len(matches) >= limit:
            break
    if matches:
        return matches

    fallback: list[str] = []
    for line_no, line in enumerate(doc.text.splitlines(), 1):
        if line.startswith("## ") or line.startswith("### PR #"):
            fallback.append(f"{doc.path.relative_to(ROOT)}:{line_no}: {line.strip()[:220]}")
        if len(fallback) >= limit:
            break
    return fallback


def _search(docs: list[Doc], query: str, limit: int) -> None:
    terms = _tokenize(query)
    if not terms:
        for doc in docs[:limit]:
            print(doc.path.relative_to(ROOT))
        return

    preferred_model = _preferred_model_slug(query)
    scored = [
        (score, doc)
        for doc in docs
        if (score := _score(doc, terms, preferred_model)) > 0
    ]
    scored.sort(key=lambda item: (-item[0], str(item[1].path)))

    for score, doc in scored[:limit]:
        print(f"## {doc.framework}/{doc.model} ({doc.path.name}) score={score}")
        print(f"Read: {doc.path.relative_to(ROOT)}")
        for line in _matching_lines(doc, terms):
            print(f"- {line}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query PR-driven model optimization history docs."
    )
    parser.add_argument("query", nargs="*", help="Search terms, model name, PR, or file path")
    parser.add_argument("--list", action="store_true", help="List framework/model slugs")
    parser.add_argument("--framework", choices=FRAMEWORKS, help="Restrict to one framework")
    parser.add_argument("--model", help="Restrict to one model-family slug")
    parser.add_argument(
        "--lang",
        choices=tuple(LANG_TO_FILES),
        default="en",
        help="Select English, Chinese, or both docs",
    )
    parser.add_argument("--paths-only", action="store_true", help="Only print matching doc paths")
    parser.add_argument("--limit", type=int, default=10, help="Maximum search results")
    args = parser.parse_args()

    if args.list:
        _list_models(args.framework)
        return 0

    docs = _docs(args.framework, args.model, args.lang)
    if args.paths_only:
        _print_paths(docs)
        return 0

    query = " ".join(args.query).strip()
    if not query and args.model:
        query = args.model
    if not query:
        parser.error("provide a query, --model, --paths-only, or --list")

    _search(docs, query, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
