from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def default_catalog_path() -> Path:
    return Path(__file__).resolve().parents[1] / "references" / "kernel-source-catalog.json"


def load_catalog(path: str | Path | None = None) -> dict[str, Any]:
    catalog_path = Path(path) if path else default_catalog_path()
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def relevant_sources(catalog: dict[str, Any], *, lane: str | None = None) -> list[dict[str, Any]]:
    lane_lower = (lane or "").lower()
    rows: list[dict[str, Any]] = []
    for source in catalog.get("local_references", []):
        rows.append(source)
    for source in catalog.get("external_repositories", []):
        if not lane_lower:
            rows.append(source)
            continue
        haystack = " ".join(
            [
                source.get("name", ""),
                " ".join(source.get("read_when", [])),
                " ".join(source.get("kernel_paths", [])),
            ]
        ).lower()
        if any(token in haystack for token in lane_lower.replace("+", " ").split()):
            rows.append(source)
    return rows


def render_catalog_for_prompt(catalog: dict[str, Any], *, lane: str | None = None) -> str:
    lines: list[str] = []
    for source in relevant_sources(catalog, lane=lane):
        name = source.get("name") or source.get("repo")
        location = source.get("path") or source.get("url")
        lines.append(f"- {name}: {location}")
        paths = source.get("must_read_first") or source.get("kernel_paths") or []
        for path in paths[:8]:
            lines.append(f"  - {path}")
    rejected = catalog.get("unconfirmed_or_rejected_aliases", [])
    if rejected:
        lines.append("- Do not use unconfirmed aliases unless revalidated:")
        for item in rejected:
            lines.append(f"  - {item.get('name')}: {item.get('status')}")
    return "\n".join(lines)
