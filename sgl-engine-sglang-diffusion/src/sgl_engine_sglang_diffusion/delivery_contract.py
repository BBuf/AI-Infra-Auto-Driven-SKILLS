"""Controller-owned, machine-readable Executor delivery contracts."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .knowledge import KnowledgeSyncError, snapshot_reference_files
from .models import (
    BaselineRecord,
    CandidateManifest,
    Delivery,
    EngagementReceipt,
    KernelEvidence,
    ResidencyEvidence,
)
from .request import FrozenBenchmarkCommand
from .techniques import TechniqueRegistry


WORKTREE_PLACEHOLDER = "{{executor_worktree}}"
DELIVERY_PLACEHOLDER = "{{delivery_path}}"


class DeliveryContractError(RuntimeError):
    """The controller cannot produce a complete delivery contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_binding(path: Path) -> dict[str, str] | None:
    return {"path": str(path), "sha256": _sha256(path)} if path.is_file() else None


def _kernelwiki_sources(campaign: Path) -> list[dict[str, str]]:
    try:
        manifest = json.loads((campaign / "KNOWLEDGE.json").read_text())
        index = Path(manifest["snapshots"]["kda_pilot"])
        references = snapshot_reference_files(
            index,
            prefix="external/KernelWiki",
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError, KnowledgeSyncError) as error:
        raise DeliveryContractError(
            f"cannot build contract without pinned KernelWiki references: {error}"
        ) from error
    return [
        {"path": str(path), "sha256": digest}
        for path, digest in sorted(references.items(), key=lambda item: str(item[0]))
    ]


def build_delivery_contract(
    *,
    campaign: Path,
    technique: str,
    registry: TechniqueRegistry,
    baseline: BaselineRecord,
    command_template: FrozenBenchmarkCommand | None,
) -> dict[str, Any]:
    """Build the exact static contract consumed by one Executor lane."""
    campaign = campaign.resolve()
    contract = registry[technique]
    profile = campaign / "profiles/0/PROFILE-DIGEST.json"
    if not profile.is_file():
        raise DeliveryContractError("active profile digest is missing")
    required = [
        "PERFORMANCE.json",
        "outputs/benchmark.jsonl",
        "outputs/media/",
        "implementation-manifest.json",
        "source-hashes.json",
        "engagement-receipt.json",
    ]
    if command_template is not None:
        required.append("COMMAND.json")
    if contract.correctness == "lossless":
        required.extend(["equivalence.json", "authenticity.json"])
    if technique == "kernel":
        required.append("KERNEL-EVIDENCE.json")
    elif technique == "residency":
        required.append("RESIDENCY-EVIDENCE.json")

    technique_schema: dict[str, Any] | None = None
    pinned_kernelwiki: list[dict[str, str]] = []
    if technique == "kernel":
        technique_schema = KernelEvidence.model_json_schema()
        pinned_kernelwiki = _kernelwiki_sources(campaign)
    elif technique == "residency":
        technique_schema = ResidencyEvidence.model_json_schema()

    command_path = campaign / "BASELINE-COMMAND.json"
    inventory_path = campaign / "GPU-INVENTORY.json"
    unavailable_inventory_path = campaign / "GPU-INVENTORY-UNAVAILABLE.json"

    return {
        "schema_version": 1,
        "technique": technique,
        "correctness": contract.correctness,
        "executor_worktree": WORKTREE_PLACEHOLDER,
        "delivery_path": DELIVERY_PLACEHOLDER,
        "baseline": {
            "path": str(campaign / "BASELINE.json"),
            "sha256": _sha256(campaign / "BASELINE.json"),
            "model_id": baseline.model_id,
            "sglang_commit": baseline.sglang_commit,
            "timing_scope": baseline.timing_scope,
            "mean_e2e_s": baseline.mean_e2e_s,
            "workload_total_s": baseline.workload_total_s,
            "request_count": baseline.request_count,
        },
        "profile": {"path": str(profile), "sha256": _sha256(profile)},
        "command_template": _artifact_binding(command_path),
        "command_template_sha256": (
            command_template.template_sha256 if command_template is not None else None
        ),
        "gpu_inventory": _artifact_binding(inventory_path),
        "gpu_inventory_unavailable": _artifact_binding(unavailable_inventory_path),
        "required_artifacts": required,
        "performance_artifact_required_fields": {
            "schema_version": 2,
            "mean_e2e_s": "positive number",
            "workload_total_s": "mean_e2e_s * request_count",
            "request_count": 5,
            "peak_memory_mib": "positive number",
            "timing_scope": baseline.timing_scope,
        },
        "schemas": {
            "delivery": Delivery.model_json_schema(),
            "candidate_manifest": CandidateManifest.model_json_schema(),
            "engagement_receipt": EngagementReceipt.model_json_schema(),
            "technique_evidence": technique_schema,
        },
        "pinned_kernelwiki_sources": pinned_kernelwiki,
        "preflight_argv": [
            sys.executable,
            "-m",
            "sgl_engine_sglang_diffusion.cli",
            "preflight-delivery",
            "--campaign",
            str(campaign),
            "--technique",
            technique,
            "--executor-worktree",
            WORKTREE_PLACEHOLDER,
            "--delivery",
            DELIVERY_PLACEHOLDER,
        ],
    }


def materialize_delivery_contract(
    value: Mapping[str, Any],
    *,
    worktree: Path,
    delivery: Path,
) -> dict[str, Any]:
    """Replace controller placeholders without interpreting arbitrary strings."""
    replacements = {
        WORKTREE_PLACEHOLDER: str(worktree.resolve()),
        DELIVERY_PLACEHOLDER: str(delivery.resolve()),
    }

    def visit(item: Any) -> Any:
        if isinstance(item, str):
            return replacements.get(item, item)
        if isinstance(item, Mapping):
            return {str(key): visit(child) for key, child in item.items()}
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            return [visit(child) for child in item]
        return item

    return visit(dict(value))
