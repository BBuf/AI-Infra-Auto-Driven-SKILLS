#!/usr/bin/env python3
"""Append one lane's immutable knowledge index to a Sol Executor seed goal."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def balanced_entries(
    entries: list[dict[str, object]], limit: int
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for entry in entries:
        groups.setdefault(str(entry["source_id"]), []).append(entry)
    for group in groups.values():
        group.sort(key=lambda item: (int(item.get("priority", 99)), str(item["path"])))
    selected = []
    source_ids = sorted(groups)
    while source_ids and len(selected) < limit:
        remaining = []
        for source_id in source_ids:
            group = groups[source_id]
            if group and len(selected) < limit:
                selected.append(group.pop(0))
            if group:
                remaining.append(source_id)
        source_ids = remaining
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--technique", required=True)
    parser.add_argument("--goal", required=True, type=Path)
    parser.add_argument("--max-entries", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_entries < 1:
        raise SystemExit("--max-entries must be positive")
    manifest_path = args.manifest.resolve()
    goal_path = args.goal.resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read knowledge manifest: {exc}") from exc
    if (
        manifest.get("schema_version") != 1
        or manifest.get("acceptance_authority") != "upstream-sol-engine"
    ):
        raise SystemExit("manifest does not preserve upstream Sol acceptance authority")
    if not goal_path.is_file():
        raise SystemExit(f"Sol seed goal does not exist: {goal_path}")

    technique = args.technique.strip()
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", technique):
        raise SystemExit(f"invalid Sol technique name: {technique!r}")
    marker = f"sol-engine-sglang-diffusion:{technique}"
    begin = f"<!-- {marker}:begin -->"
    end = f"<!-- {marker}:end -->"
    original = goal_path.read_text(encoding="utf-8")
    if begin in original or end in original:
        raise SystemExit(f"knowledge block is already present for {technique}")

    entries = [
        entry
        for entry in manifest.get("entries", [])
        if isinstance(entry, dict) and technique in entry.get("eligible_techniques", [])
    ]
    if not entries:
        raise SystemExit(
            f"manifest has no knowledge eligible for technique {technique!r}"
        )
    selected = balanced_entries(entries, args.max_entries)
    sources = {source["id"]: source for source in manifest.get("sources", [])}
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    lines = [
        begin,
        f"## SGLang Diffusion knowledge for `{technique}`",
        "",
        f"Manifest: `{manifest_path}` (sha256 `{manifest_sha}`).",
        "",
        "These immutable sources suggest hypotheses only. Do not use historical",
        "speedups, tolerances, GPU thresholds, or prior judgments as acceptance",
        "criteria. The appended upstream technique scope, loop contract, frozen",
        "baseline, verifier, Master review, and quality rules remain authoritative.",
        "",
        "Read the relevant source before using an idea and cite its source id,",
        "commit, relative path, and digest in the delivery evidence.",
        "",
        "### Frozen sources",
        "",
    ]
    for source_id in sorted({entry["source_id"] for entry in entries}):
        source = sources[source_id]
        lines.append(f"- `{source_id}`: `{source['commit']}` at `{source['root']}`")
    lines.extend(["", "### Eligible references", ""])
    for entry in selected:
        lines.append(
            f"- `{entry['source_id']}:{entry['path']}` " f"(sha256 `{entry['sha256']}`)"
        )
    if len(entries) > len(selected):
        lines.append(
            f"- … inspect the manifest for {len(entries) - len(selected)} more "
            "eligible references."
        )
    history = manifest.get("sglang_history", [])
    if history:
        lines.extend(
            [
                "",
                "### Recent SGLang Diffusion commits",
                "",
                "Inspect diffs with `git show`; subjects alone are not evidence.",
                "",
            ]
        )
        for item in history[:30]:
            lines.append(f"- `{item['commit']}` — {item['subject']}")
    lines.extend(["", end, ""])

    separator = "" if original.endswith("\n") else "\n"
    goal_path.write_text(original + separator + "\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "goal": str(goal_path),
                "technique": technique,
                "entry_count": len(selected),
                "manifest_sha256": manifest_sha,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
