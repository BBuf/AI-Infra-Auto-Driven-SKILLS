"""Allowlisted, provenance-preserving optimization knowledge snapshots."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

MAX_KNOWLEDGE_FILE_BYTES = 2 * 1024 * 1024
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
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?im)\b("
    r"HF_TOKEN|HUGGING_FACE_HUB_TOKEN|GITHUB_TOKEN|GH_TOKEN|"
    r"API_KEY|ACCESS_TOKEN|SECRET|PASSWORD"
    r")\b(\s*[:=]\s*)([^\s`\"']+)"
)
_PRIVATE_PATH_RE = re.compile(r"(?<![\w.-])/(?:Users|home)/[^/\s`\"']+(?:/[^\s`\"']*)?")
_MARKDOWN_HEADING_RE = re.compile(r"(?m)^#{1,6}\s+(.+?)\s*$")
_DECLARATION_RE = re.compile(
    r"(?m)^\s*(?:async\s+def|def|class|struct|enum|namespace)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)"
)
_CALLABLE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_:]{2,})\s*\(")


class KnowledgeSyncError(RuntimeError):
    """Knowledge could not be snapshotted without violating the contract."""


@dataclass(frozen=True)
class KnowledgeEntry:
    path: str
    media_type: str
    sha256: str
    reference_sha256: str
    headings: tuple[str, ...]
    symbols: tuple[str, ...]
    executable: bool = False


@dataclass(frozen=True)
class KnowledgeSnapshot:
    schema_version: int
    source: str
    commit: str
    entries: tuple[KnowledgeEntry, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "commit": self.commit,
            "entries": [asdict(entry) for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "KnowledgeSnapshot":
        raw_entries = data.get("entries")
        if not isinstance(raw_entries, list) or not raw_entries:
            raise KnowledgeSyncError(
                "knowledge index entries must be a nonempty list"
            )
        entries = tuple(
            KnowledgeEntry(
                path=str(item["path"]),
                media_type=str(item["media_type"]),
                sha256=str(item["sha256"]),
                reference_sha256=str(item["reference_sha256"]),
                headings=tuple(item.get("headings", [])),
                symbols=tuple(item.get("symbols", [])),
                executable=bool(item.get("executable", False)),
            )
            for item in raw_entries
            if isinstance(item, dict)
        )
        if len(entries) != len(raw_entries):
            raise KnowledgeSyncError("knowledge index contains a malformed entry")
        return cls(
            schema_version=int(data["schema_version"]),
            source=str(data["source"]),
            commit=str(data["commit"]),
            entries=entries,
        )


def load_registry(path: Path) -> dict[str, list[str]]:
    """Load source path allowlists from the checked-in TOML registry."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise KnowledgeSyncError("unsupported knowledge registry schema")
    raw_sources = data.get("sources")
    if not isinstance(raw_sources, dict):
        raise KnowledgeSyncError("knowledge registry is missing [sources]")

    sources: dict[str, list[str]] = {}
    for name, raw in raw_sources.items():
        if not isinstance(raw, dict) or not isinstance(raw.get("paths"), list):
            raise KnowledgeSyncError(f"source {name!r} has no path allowlist")
        patterns = [str(pattern) for pattern in raw["paths"]]
        if not patterns or any(Path(pattern).is_absolute() for pattern in patterns):
            raise KnowledgeSyncError(f"source {name!r} has an invalid path allowlist")
        sources[str(name)] = patterns
    return sources


def sanitize_text(text: str) -> str:
    """Redact credential assignments and machine-specific home paths."""

    def redact_secret(match: re.Match[str]) -> str:
        return f"{match.group(1)}{match.group(2)}<redacted>"

    text = _SECRET_ASSIGNMENT_RE.sub(redact_secret, text)
    return _PRIVATE_PATH_RE.sub("<redacted-absolute-path>", text)


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return "text/markdown"
    if suffix in {".json"}:
        return "application/json"
    if suffix in {".toml"}:
        return "application/toml"
    if suffix in {".yaml", ".yml"}:
        return "application/yaml"
    if suffix in {".py", ".sh"}:
        return "text/x-script"
    if suffix in {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}:
        return "text/x-source"
    return "text/plain"


def _extract_metadata(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    headings = tuple(
        sorted({heading.strip() for heading in _MARKDOWN_HEADING_RE.findall(text)})
    )
    symbols = set(_DECLARATION_RE.findall(text))
    symbols.update(_CALLABLE_RE.findall(text))
    return headings, tuple(sorted(symbols)[:512])


def _iter_allowed_files(checkout: Path, patterns: Iterable[str]) -> list[Path]:
    root = checkout.resolve()
    matched: set[Path] = set()
    for pattern in patterns:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise KnowledgeSyncError(f"unsafe allowlist pattern: {pattern}")
        for candidate in root.glob(pattern):
            candidates = candidate.rglob("*") if candidate.is_dir() else (candidate,)
            for path in candidates:
                resolved = path.resolve()
                try:
                    resolved.relative_to(root)
                except ValueError as exc:
                    raise KnowledgeSyncError(
                        f"allowlisted path escapes checkout: {path}"
                    ) from exc
                if (
                    resolved.is_file()
                    and not resolved.is_symlink()
                    and resolved.suffix.lower() in TEXT_SUFFIXES
                ):
                    matched.add(resolved)
    return sorted(matched, key=lambda path: path.relative_to(root).as_posix())


def _load_existing_snapshot(
    index_path: Path, *, source: str, commit: str
) -> KnowledgeSnapshot | None:
    if not index_path.is_file():
        return None
    data = json.loads(index_path.read_text(encoding="utf-8"))
    snapshot = KnowledgeSnapshot.from_dict(data)
    if snapshot.source != source or snapshot.commit != commit:
        raise KnowledgeSyncError(
            "knowledge output already belongs to a different source revision"
        )
    return snapshot


def sync_source(
    *,
    name: str,
    checkout: Path,
    commit: str,
    patterns: Iterable[str],
    output_dir: Path,
) -> KnowledgeSnapshot:
    """Create an immutable text-only knowledge snapshot for one locked source."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise KnowledgeSyncError("knowledge source commit must be a full SHA")
    checkout = checkout.resolve()
    if not checkout.is_dir():
        raise KnowledgeSyncError(f"knowledge checkout does not exist: {checkout}")

    index_path = output_dir / "index.json"
    existing = _load_existing_snapshot(index_path, source=name, commit=commit)
    if existing is not None:
        return existing
    if output_dir.exists() and any(output_dir.iterdir()):
        raise KnowledgeSyncError(
            f"knowledge output is nonempty without a valid index: {output_dir}"
        )

    references_dir = output_dir / "references"
    references_dir.mkdir(parents=True, exist_ok=True)
    entries: list[KnowledgeEntry] = []
    for source_path in _iter_allowed_files(checkout, patterns):
        raw = source_path.read_bytes()
        if len(raw) > MAX_KNOWLEDGE_FILE_BYTES:
            raise KnowledgeSyncError(
                f"knowledge file exceeds {MAX_KNOWLEDGE_FILE_BYTES} bytes: "
                f"{source_path.relative_to(checkout)}"
            )
        relative = source_path.relative_to(checkout)
        text = sanitize_text(raw.decode("utf-8", errors="replace"))
        reference = references_dir / relative
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.write_text(text, encoding="utf-8")
        headings, symbols = _extract_metadata(text)
        entries.append(
            KnowledgeEntry(
                path=relative.as_posix(),
                media_type=_media_type(source_path),
                sha256=hashlib.sha256(raw).hexdigest(),
                reference_sha256=hashlib.sha256(text.encode()).hexdigest(),
                headings=headings,
                symbols=symbols,
            )
        )

    if not entries:
        raise KnowledgeSyncError(
            f"knowledge source {name!r} matched no allowlisted text files"
        )

    snapshot = KnowledgeSnapshot(
        schema_version=1,
        source=name,
        commit=commit,
        entries=tuple(entries),
    )
    temporary_index = output_dir / ".index.json.tmp"
    temporary_index.write_text(
        json.dumps(snapshot.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_index.replace(index_path)
    return snapshot


def compute_contract_hashes(
    checkout: Path, authoritative_paths: Iterable[str]
) -> dict[str, str]:
    """Hash reviewed Sol-Engine contract inputs without copying their content."""
    root = checkout.resolve()
    result: dict[str, str] = {}
    for relative_text in sorted(set(authoritative_paths)):
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise KnowledgeSyncError(f"unsafe contract path: {relative}")
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise KnowledgeSyncError(
                f"contract path escapes checkout: {relative}"
            ) from exc
        if not source.is_file():
            raise KnowledgeSyncError(f"missing Sol-Engine contract: {relative}")
        result[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    return result


def read_source_lock(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"repository", "commit", "authoritative_paths"}
    if not required.issubset(data):
        missing = sorted(required - set(data))
        raise KnowledgeSyncError(f"source lock is missing: {missing}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(data["commit"])):
        raise KnowledgeSyncError("Sol-Engine source lock must use a full commit")
    if not isinstance(data["authoritative_paths"], list):
        raise KnowledgeSyncError("authoritative_paths must be a list")
    return data


def write_contract_hashes(
    source_lock_path: Path, checkout: Path, output_path: Path
) -> dict[str, str]:
    lock = read_source_lock(source_lock_path)
    hashes = compute_contract_hashes(checkout, lock["authoritative_paths"])
    payload = {
        "schema_version": 1,
        "repository": lock["repository"],
        "commit": lock["commit"],
        "hashes": hashes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    return hashes


def check_contract_hashes(
    source_lock_path: Path, checkout: Path, expected_path: Path
) -> list[str]:
    lock = read_source_lock(source_lock_path)
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    actual = compute_contract_hashes(checkout, lock["authoritative_paths"])
    issues: list[str] = []
    if expected.get("commit") != lock["commit"]:
        issues.append("contract hash commit does not match source lock")
    expected_hashes = expected.get("hashes", {})
    for path in sorted(set(actual) | set(expected_hashes)):
        if actual.get(path) != expected_hashes.get(path):
            issues.append(f"contract drift: {path}")
    return issues
