#!/usr/bin/env python3
"""Build a deterministic, read-only SGLang Diffusion knowledge manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".json",
    ".md",
    ".mu",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class Source:
    source_id: str
    root: Path
    include: Callable[[str], bool]


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SystemExit(f"git {' '.join(args)} failed for {root}: {detail}")
    return result.stdout


def validate_source(source: Source) -> str:
    root = source.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"knowledge source is not a directory: {root}")
    top = Path(git(root, "rev-parse", "--show-toplevel").strip()).resolve()
    if top != root:
        raise SystemExit(f"knowledge source must be a Git root: {root} (root is {top})")
    commit = git(root, "rev-parse", "HEAD^{commit}").strip()
    if not FULL_SHA.fullmatch(commit):
        raise SystemExit(f"knowledge source has no full frozen commit: {root}")
    dirty = git(root, "status", "--porcelain", "--untracked-files=no").strip()
    if dirty:
        raise SystemExit(f"knowledge source has tracked changes: {root}\n{dirty}")
    return commit


def tracked_files(source: Source) -> list[Path]:
    output = subprocess.run(
        ["git", "-C", str(source.root), "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    files = []
    for raw in output.split(b"\0"):
        if not raw:
            continue
        relative = raw.decode("utf-8", errors="strict")
        path = source.root / relative
        if (
            source.include(relative)
            and path.is_file()
            and path.suffix.lower() in TEXT_SUFFIXES
        ):
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(source.root).as_posix())


def sglang_include(path: str) -> bool:
    lower = path.lower()
    return lower.startswith("python/sglang/multimodal_gen/") or (
        "diffusion" in lower
        and lower.startswith(("docs/", "test/", "tests/", "sgl-kernel/"))
    )


def route(path: str) -> tuple[str, list[str], str]:
    lower = path.lower().replace("-", "_")
    if any(
        token in lower
        for token in ("quant", "nvfp4", "fp8", "int8", "nunchaku", "modelopt")
    ):
        return (
            "quantization",
            [],
            "knowledge_only_outside_current_sol_registry",
        )
    if any(token in lower for token in ("cache", "teacache", "magcache", "fbcache")):
        return "cache", ["cache"], "hypothesis_only"
    if any(
        token in lower
        for token in (
            "pisa",
            "sparse",
            "token_prun",
            "token_select",
            "vmoba",
            "attention_pa",
        )
    ):
        return "sparse_attention", ["pisa"], "hypothesis_only"
    if any(
        token in lower
        for token in (
            "topology",
            "distributed",
            "ulysses",
            "sequence_parallel",
            "context_parallel",
            "all_to_all",
            "collective",
        )
    ):
        return "distributed_topology", ["topology"], "hypothesis_only"
    return "lossless_kernel", ["kernel"], "hypothesis_only"


def entry_priority(path: str) -> int:
    lower = path.lower()
    if "/apps/" in lower:
        return 6
    if lower.endswith("readme.md") or "/docs/" in lower:
        return 1
    if any(
        token in lower
        for token in (
            "/runtime/layers/",
            "/attention/",
            "/distributed/",
            "/csrc/",
            "/kernel",
            "cache",
        )
    ):
        return 2
    if "/configs/" in lower:
        return 3
    if any(token in lower for token in ("/test", "/bench")):
        return 5
    return 4


def balanced_entries(
    entries: list[dict[str, object]], limit: int
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for entry in entries:
        groups.setdefault(str(entry["source_id"]), []).append(entry)
    for group in groups.values():
        group.sort(key=lambda item: (int(item["priority"]), str(item["path"])))
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


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def sglang_history(root: Path) -> list[dict[str, str]]:
    raw = git(
        root,
        "log",
        "--max-count=200",
        "--format=%H%x00%s",
        "--",
        "python/sglang/multimodal_gen",
    )
    history = []
    for line in raw.splitlines():
        commit, separator, subject = line.partition("\0")
        if separator and FULL_SHA.fullmatch(commit):
            history.append({"commit": commit, "subject": subject})
    return history


def render_index(manifest: dict[str, object]) -> str:
    sources = manifest["sources"]
    entries = manifest["entries"]
    assert isinstance(sources, list)
    assert isinstance(entries, list)
    lines = [
        "# SGLang Diffusion Executor Knowledge Index",
        "",
        "This index is hypothesis material only. Upstream Sol Engine remains the",
        "acceptance authority for correctness, quality, performance, integration,",
        "and termination. Historical speedups and thresholds are never gates.",
        "",
        "## Frozen sources",
        "",
    ]
    for source in sources:
        assert isinstance(source, dict)
        lines.append(f"- `{source['id']}`: `{source['commit']}` at `{source['root']}`")
    for technique in ("kernel", "cache", "pisa", "topology"):
        selected = [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and technique in entry.get("eligible_techniques", [])
        ]
        displayed = balanced_entries(selected, 80)
        lines.extend(["", f"## {technique}", ""])
        for entry in displayed:
            lines.append(
                f"- `{entry['source_id']}:{entry['path']}` "
                f"(sha256 `{entry['sha256']}`)"
            )
        if len(selected) > len(displayed):
            lines.append(
                f"- … {len(selected) - len(displayed)} more entries are available in the manifest."
            )
    excluded = [
        entry
        for entry in entries
        if isinstance(entry, dict) and not entry.get("eligible_techniques")
    ]
    lines.extend(
        [
            "",
            "## Knowledge-only outside the pinned Sol registry",
            "",
            "Do not inject these into a lossless lane. They become eligible only if",
            "the pinned upstream Sol revision registers a compatible quality-gated lane.",
            "",
        ]
    )
    for entry in excluded[:80]:
        lines.append(f"- `{entry['source_id']}:{entry['path']}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kda-root", required=True, type=Path)
    parser.add_argument("--sglang-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kda = args.kda_root.resolve()
    sglang = args.sglang_root.resolve()
    sources = [
        Source("kda-pilot", kda, lambda path: path.startswith("diffusion/")),
        Source("kernelwiki", kda / "external" / "KernelWiki", lambda path: True),
        Source(
            "ncu-report-skill",
            kda / "external" / "ncu-report-skill",
            lambda path: True,
        ),
        Source(
            "warp-specialization-report-skill",
            kda / "external" / "warp-specialization-report-skill",
            lambda path: True,
        ),
        Source("sglang-main", sglang, sglang_include),
    ]
    source_rows = []
    entries = []
    for source in sources:
        commit = validate_source(source)
        files = tracked_files(source)
        if not files:
            raise SystemExit(f"knowledge source selected no files: {source.root}")
        source_rows.append(
            {
                "id": source.source_id,
                "root": str(source.root.resolve()),
                "commit": commit,
            }
        )
        for path in files:
            relative = path.relative_to(source.root).as_posix()
            topic, techniques, status = route(relative)
            entries.append(
                {
                    "source_id": source.source_id,
                    "path": relative,
                    "sha256": digest(path),
                    "bytes": path.stat().st_size,
                    "topic": topic,
                    "eligible_techniques": techniques,
                    "status": status,
                    "priority": entry_priority(relative),
                }
            )
    entries.sort(key=lambda entry: (entry["source_id"], entry["path"]))
    manifest = {
        "schema_version": 1,
        "acceptance_authority": "upstream-sol-engine",
        "historical_evidence_policy": "hypothesis_only_never_an_acceptance_gate",
        "sources": source_rows,
        "entries": entries,
        "sglang_history": sglang_history(sglang),
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "KNOWLEDGE-MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    index_path = output / "EXECUTOR-KNOWLEDGE.md"
    index_path.write_text(render_index(manifest), encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "index": str(index_path),
                "entry_count": len(entries),
                "manifest_sha256": digest(manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
