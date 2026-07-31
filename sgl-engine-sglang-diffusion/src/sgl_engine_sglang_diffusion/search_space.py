"""Build a deterministic, provenance-rich view of Sol's optimization space."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class SearchSpaceError(RuntimeError):
    """The locked Sol checkout cannot produce a complete search-space catalog."""


_FAMILY_DOCUMENTS = {
    "cache": "search_space/01_cache.md",
    "token_pruning": "search_space/02_token_pruning.md",
    "quantization": "search_space/03_quantization.md",
    "sparse_attention": "search_space/04_sparse_attention.md",
    "kernel": "search_space/05_kernel_fusion.md",
    "topology": "search_space/06_parallel_topology.md",
}
_DIMENSION_FAMILIES = {
    "kwl_fusion": "kernel",
    "step_cache": "cache",
    "sparse_attention": "sparse_attention",
    "nvfp4_ffn": "quantization",
    "token_prune": "token_pruning",
}
_SITE_DOC_FAMILIES = {
    "cache": "cache",
    "kernel": "kernel",
    "quant": "quantization",
    "sparse": "sparse_attention",
    "token_prune": "token_pruning",
}
_REGISTER_RE = re.compile(
    r"""@register_(technique|transform)\(\s*["']([^"']+)["']"""
)
_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    value = re.sub(r"^\d+\.\s*", "", value.strip())
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not value:
        raise SearchSpaceError("method-family title has no stable identifier")
    return value


def _method_families(document: Path) -> list[dict[str, str]]:
    text = document.read_text(encoding="utf-8")
    lines = text.splitlines()
    start: int | None = None
    for index, line in enumerate(lines):
        if re.fullmatch(r"##\s+Method [Ff]amilies", line.strip()):
            start = index + 1
            break
    if start is None:
        raise SearchSpaceError(f"missing Method Families section: {document}")

    section: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("## "):
            break
        section.append(stripped)

    headings = [
        line.removeprefix("### ").strip()
        for line in section
        if line.startswith("### ")
    ]
    raw_titles = (
        headings
        if headings
        else [
            line.removeprefix("- ").split(":", 1)[0].strip()
            for line in section
            if line.startswith("- ")
        ]
    )
    methods: list[dict[str, str]] = []
    for raw_title in raw_titles:
        title = re.sub(r"^\d+\.\s*", "", raw_title)
        method_id = _slug(title)
        if method_id not in {item["id"] for item in methods}:
            methods.append(
                {
                    "id": method_id,
                    "title": title,
                    "coverage_status": "documented",
                }
            )
    if not methods:
        raise SearchSpaceError(f"no method families found in {document}")
    return methods


def _site_documents(root: Path) -> dict[str, list[dict[str, str]]]:
    result = {family: [] for family in _FAMILY_DOCUMENTS}
    site_root = root / "site_docs" / "techniques"
    if not site_root.is_dir():
        return result
    for path in sorted(site_root.rglob("*.md")):
        relative = path.relative_to(root).as_posix()
        stem = path.relative_to(site_root).parts[0]
        stem = Path(stem).stem
        family = _SITE_DOC_FAMILIES.get(stem)
        if family is None:
            continue
        result[family].append({"path": relative, "sha256": _sha256(path)})
    return result


def _structured_candidate(
    *,
    root: Path,
    path: Path,
    data: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    identity = data["id"]
    assert isinstance(identity, dict)
    candidate_id = str(identity.get("name", "")).strip()
    dimension = str(identity.get("dimension", "")).strip()
    candidate_family = str(identity.get("family", "")).strip()
    if not candidate_id or not dimension or not candidate_family:
        raise SearchSpaceError(f"incomplete structured candidate identity: {path}")
    family = _DIMENSION_FAMILIES.get(dimension)
    if family is None:
        raise SearchSpaceError(
            f"unknown Sol candidate dimension {dimension!r}: "
            f"{path.relative_to(root)}"
        )

    references = data.get("references")
    local = references.get("local") if isinstance(references, dict) else None
    generic_impl = (
        str(local.get("generic_impl", "")).strip()
        if isinstance(local, dict)
        else ""
    )
    if not generic_impl:
        raise SearchSpaceError(
            f"structured candidate {candidate_id!r} has no generic implementation"
        )
    implementation = root / generic_impl
    if not implementation.is_file():
        raise SearchSpaceError(
            f"structured candidate {candidate_id!r} references missing "
            f"implementation {generic_impl!r}"
        )

    requirements = data.get("requirements")
    capabilities = (
        requirements.get("capabilities")
        if isinstance(requirements, dict)
        else None
    )
    if not isinstance(capabilities, list) or not capabilities:
        raise SearchSpaceError(
            f"structured candidate {candidate_id!r} has no capabilities"
        )
    efficiency = data.get("efficiency")
    verification = data.get("verification")
    if not isinstance(efficiency, dict) or not isinstance(verification, dict):
        raise SearchSpaceError(
            f"structured candidate {candidate_id!r} lacks execution metadata"
        )

    return family, {
        "id": candidate_id,
        "coverage_status": "referenced",
        "dimension": dimension,
        "candidate_family": candidate_family,
        "kind": str(data.get("kind", "")),
        "purpose": str(data.get("purpose", "")),
        "description": str(data.get("description", "")),
        "model_profile": str(data.get("model_profile", "")),
        "required_capabilities": sorted(str(item) for item in capabilities),
        "efficiency_kind": str(efficiency.get("kind", "")),
        "efficiency_name": str(efficiency.get("name", "")),
        "generic_impl": generic_impl,
        "generic_impl_sha256": _sha256(implementation),
        "quality_gate": str(verification.get("quality_gate", "")),
        "source_path": path.relative_to(root).as_posix(),
        "source_sha256": _sha256(path),
    }


def _candidate_catalog(
    root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    candidates = {family: [] for family in _FAMILY_DOCUMENTS}
    recipes: list[dict[str, Any]] = []
    candidates_root = root / "candidates"
    if not candidates_root.is_dir():
        raise SearchSpaceError("locked Sol checkout has no candidates directory")
    for path in sorted(candidates_root.rglob("*.toml")):
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
            raise SearchSpaceError(f"invalid Sol candidate manifest {path}: {error}")
        identity = data.get("id")
        if isinstance(identity, dict):
            family, candidate = _structured_candidate(
                root=root, path=path, data=data
            )
            candidates[family].append(candidate)
        elif isinstance(identity, str) and identity.strip():
            recipes.append(
                {
                    "id": identity.strip(),
                    "coverage_status": "referenced",
                    "kind": str(data.get("kind", "")),
                    "purpose": str(data.get("purpose", "")),
                    "description": str(data.get("description", "")),
                    "model_profile": str(data.get("model_profile", "")),
                    "source_path": path.relative_to(root).as_posix(),
                    "source_sha256": _sha256(path),
                }
            )
    return candidates, recipes


def _registered_implementations(root: Path) -> list[dict[str, str]]:
    technique_root = root / "techniques"
    if not technique_root.is_dir():
        raise SearchSpaceError("locked Sol checkout has no techniques directory")
    implementations: list[dict[str, str]] = []
    for path in sorted(technique_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for kind, name in _REGISTER_RE.findall(text):
            implementations.append(
                {
                    "kind": kind,
                    "name": name,
                    "coverage_status": "referenced",
                    "source_path": path.relative_to(root).as_posix(),
                    "source_sha256": _sha256(path),
                }
            )
    return implementations


def build_sol_search_space_catalog(
    *,
    sol_checkout: Path,
    sol_commit: str,
    output_path: Path,
) -> dict[str, Any]:
    """Build and atomically persist the complete reviewed Sol opportunity index."""
    if not _SHA40_RE.fullmatch(sol_commit):
        raise SearchSpaceError("Sol search-space catalog requires a full commit")
    root = sol_checkout.resolve()
    if not root.is_dir():
        raise SearchSpaceError(f"Sol checkout does not exist: {root}")

    candidates, recipes = _candidate_catalog(root)
    site_documents = _site_documents(root)
    families: dict[str, dict[str, Any]] = {}
    for family, relative_text in _FAMILY_DOCUMENTS.items():
        document = root / relative_text
        if not document.is_file():
            raise SearchSpaceError(
                f"missing canonical Sol search document: {relative_text}"
            )
        methods = _method_families(document)
        family_candidates = sorted(candidates[family], key=lambda item: item["id"])
        review_items = [
            *[f"method:{item['id']}" for item in methods],
            *[f"candidate:{item['id']}" for item in family_candidates],
        ]
        families[family] = {
            "document": relative_text,
            "document_sha256": _sha256(document),
            "methods": methods,
            "site_documents": site_documents[family],
            "candidates": family_candidates,
            "review_items": sorted(set(review_items)),
        }

    implementations = _registered_implementations(root)
    registered = {(item["kind"], item["name"]) for item in implementations}
    for value in families.values():
        for candidate in value["candidates"]:
            expected_kind = (
                "transform"
                if candidate["efficiency_kind"] in {"build_transform", "transform"}
                else "technique"
            )
            candidate["implementation_registered"] = (
                expected_kind,
                candidate["efficiency_name"],
            ) in registered

    payload: dict[str, Any] = {
        "schema_version": 1,
        "source": "sol_engine",
        "sol_commit": sol_commit,
        "families": families,
        "candidate_count": sum(
            len(value["candidates"]) for value in families.values()
        ),
        "recipes": sorted(recipes, key=lambda item: item["id"]),
        "recipe_count": len(recipes),
        "implementations": sorted(
            implementations,
            key=lambda item: (item["kind"], item["name"], item["source_path"]),
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return payload
