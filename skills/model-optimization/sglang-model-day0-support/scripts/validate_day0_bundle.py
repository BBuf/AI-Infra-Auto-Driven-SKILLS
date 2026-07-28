#!/usr/bin/env python3
"""Validate a resolved, public SGLang model Day-0 support bundle."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

REQUIRED_FILES = {
    "scope-contract.md": (
        "# Day-0 Scope Contract",
        "## Release Cut",
        "## Required Capabilities",
        "## Out of Scope",
    ),
    "architecture-gap-map.md": (
        "# Architecture Gap Map",
        "## Capability Classification",
        "## Evidence",
    ),
    "pr-dag.md": (
        "# Pull Request DAG",
        "## Dependencies",
        "## Merge Gates",
    ),
    "validation-matrix.md": (
        "# Validation Matrix",
        "## Risk Pairs",
        "## Required Lanes",
    ),
    "release-lock.md": (
        "# Release Lock",
        "## Source Revisions",
        "## Artifacts",
        "## Limitations",
    ),
    "pr-body.md": (
        "# Public Pull Request Body",
        "## Summary",
        "## Validation",
        "## Limitations",
        "## Evidence",
    ),
    "follow-up-ledger.md": (
        "# Post-Day-0 Follow-up Ledger",
        "## Open Fixes",
        "## Performance Work",
        "## Experiments and Reverts",
    ),
    "sanitization-report.md": (
        "# Sanitization Report",
        "## Public Evidence",
        "## Denylist Result",
    ),
}

PLACEHOLDER_PATTERN = re.compile(r"\{\{[^{}\n]+\}\}")
UNRESOLVED_WORD_PATTERN = re.compile(r"\b(?:TBD|TODO)\b", re.IGNORECASE)
PR_URL_PATTERN = re.compile(
    r"https://github\.com/" r"([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)/?"
)
EVIDENCE_PATTERN = re.compile(
    r"^- Evidence:\s+"
    r"(https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/"
    r"pull/[1-9][0-9]*/?)"
    r"(?P<fields>.*)$",
    re.MULTILINE,
)
PRIVATE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])/(?:Users|home|data|workspace|mnt)/[^\s`)\]]+"
    r"|[A-Za-z]:\\Users\\[^\s`)\]]+"
)
IPV4_PATTERN = re.compile(r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])")
SSH_GIT_PATTERN = re.compile(r"(?:git" + r"@|ssh://)", re.IGNORECASE)
SECRET_PATTERNS = (
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    re.compile(
        r"\b(?:password|passwd|api[_-]?key|access[_-]?token)" r"\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
)
VALID_EVIDENCE_STATES = {"open", "merged", "closed", "reverted"}


def _finding(filename: str, message: str) -> str:
    return f"{filename}: {message}"


def _valid_ipv4(value: str) -> bool:
    return all(int(part) <= 255 for part in value.split("."))


def _parse_evidence_fields(raw_fields: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for segment in raw_fields.split("|"):
        segment = segment.strip()
        if not segment or ":" not in segment:
            continue
        key, value = segment.split(":", 1)
        fields[key.strip().lower()] = value.strip()
    return fields


def _validate_evidence(
    filename: str,
    text: str,
    allowed_repositories: set[str],
) -> list[str]:
    findings: list[str] = []
    evidence_starts = [
        line for line in text.splitlines() if line.startswith("- Evidence:")
    ]
    matches = list(EVIDENCE_PATTERN.finditer(text))
    if len(evidence_starts) != len(matches):
        findings.append(
            _finding(filename, "malformed evidence record; use the canonical syntax")
        )

    for match in matches:
        url = match.group(1).rstrip("/")
        pr_match = PR_URL_PATTERN.fullmatch(url)
        if pr_match is None:
            findings.append(_finding(filename, f"invalid evidence PR URL: {url}"))
            continue
        repository = f"{pr_match.group(1)}/{pr_match.group(2)}"
        if repository not in allowed_repositories:
            findings.append(
                _finding(filename, f"repository is not allowed: {repository}")
            )

        fields = _parse_evidence_fields(match.group("fields"))
        state = fields.get("state", "").lower()
        head = fields.get("head", "")
        limitation = fields.get("limitation", "")

        if state not in VALID_EVIDENCE_STATES:
            findings.append(_finding(filename, "evidence requires a valid state"))

        if not re.fullmatch(r"[0-9a-fA-F]{40}", head):
            if state == "open":
                findings.append(
                    _finding(filename, "open evidence requires immutable head")
                )
            else:
                findings.append(
                    _finding(filename, "evidence requires a 40-hex immutable head")
                )

        if not limitation:
            if state == "open":
                findings.append(_finding(filename, "open evidence requires limitation"))
            else:
                findings.append(_finding(filename, "evidence requires limitation"))
    return findings


def _validate_text(
    filename: str,
    text: str,
    required_headings: Sequence[str],
    allowed_repositories: set[str],
    denylist: Sequence[str],
) -> list[str]:
    findings: list[str] = []
    lines = set(text.splitlines())
    for heading in required_headings:
        if heading not in lines:
            findings.append(_finding(filename, f"missing required heading: {heading}"))

    for match in PLACEHOLDER_PATTERN.finditer(text):
        findings.append(_finding(filename, f"unresolved placeholder: {match.group(0)}"))
    for match in UNRESOLVED_WORD_PATTERN.finditer(text):
        findings.append(_finding(filename, f"unresolved marker: {match.group(0)}"))

    for match in PR_URL_PATTERN.finditer(text):
        repository = f"{match.group(1)}/{match.group(2)}"
        if repository not in allowed_repositories:
            findings.append(
                _finding(filename, f"repository is not allowed: {repository}")
            )

    if PRIVATE_PATH_PATTERN.search(text):
        findings.append(_finding(filename, "absolute private path detected"))

    if any(_valid_ipv4(match.group(0)) for match in IPV4_PATTERN.finditer(text)):
        findings.append(_finding(filename, "IP address detected"))

    if SSH_GIT_PATTERN.search(text):
        findings.append(_finding(filename, "SSH Git URL detected"))

    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            findings.append(_finding(filename, "secret-like value detected"))
            break

    for entry in denylist:
        if entry and entry in text:
            findings.append(_finding(filename, f"denylist entry detected: {entry}"))

    findings.extend(_validate_evidence(filename, text, allowed_repositories))
    return findings


def validate_bundle(
    root: Path,
    allowed_repositories: set[str],
    denylist: Sequence[str],
) -> list[str]:
    """Return all independent findings for a Day-0 bundle."""
    root = Path(root)
    if not root.is_dir():
        return [f"bundle directory does not exist: {root}"]

    findings: list[str] = []
    for filename, required_headings in REQUIRED_FILES.items():
        path = root / filename
        if not path.is_file():
            findings.append(f"missing required file: {filename}")
            continue
        text = path.read_text(encoding="utf-8")
        findings.extend(
            _validate_text(
                filename,
                text,
                required_headings,
                allowed_repositories,
                denylist,
            )
        )
    return findings


def _read_denylist(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a resolved public SGLang model Day-0 bundle."
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--allow-repository",
        action="append",
        dest="allowed_repositories",
        help="Allow a GitHub PR owner/repository; defaults to sgl-project/sglang",
    )
    parser.add_argument(
        "--denylist",
        type=Path,
        help="Uncommitted file containing one forbidden literal per line",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        dest="output_format",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    allowed = set(args.allowed_repositories or {"sgl-project/sglang"})
    findings = validate_bundle(
        args.bundle,
        allowed,
        _read_denylist(args.denylist),
    )
    if args.output_format == "json":
        print(json.dumps({"valid": not findings, "findings": findings}, indent=2))
    elif findings:
        print("Day-0 bundle is invalid:")
        for finding in findings:
            print(f"- {finding}")
    else:
        print("Day-0 bundle is valid.")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
