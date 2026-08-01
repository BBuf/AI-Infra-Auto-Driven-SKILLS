from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sgl_engine_sglang_diffusion.delivery_contract import (
    build_delivery_contract,
    materialize_delivery_contract,
)
from sgl_engine_sglang_diffusion.models import BaselineRecord
from sgl_engine_sglang_diffusion.techniques import TechniqueRegistry


def test_kernel_contract_exposes_exact_pinned_citations_and_schemas(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    profile = campaign / "profiles/0/PROFILE-DIGEST.json"
    profile.parent.mkdir(parents=True)
    profile.write_text('{"profile":"bound"}\n')
    baseline_path = campaign / "BASELINE.json"
    baseline_path.write_text('{"baseline":"bound"}\n')
    inventory_path = campaign / "GPU-INVENTORY.json"
    inventory_path.write_text('{"gpu_count":1}\n')
    reference = (
        campaign
        / "knowledge/kda_pilot/locked/references/external/KernelWiki/wiki/fusion.md"
    )
    reference.parent.mkdir(parents=True)
    reference.write_text("# Fusion\n")
    digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    index = campaign / "knowledge/kda_pilot/locked/index.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "kda_pilot",
                "commit": "d" * 40,
                "entries": [
                    {
                        "path": "external/KernelWiki/wiki/fusion.md",
                        "media_type": "text/markdown",
                        "sha256": digest,
                        "reference_sha256": digest,
                        "headings": ["Fusion"],
                        "symbols": [],
                        "executable": False,
                    }
                ],
            }
        )
    )
    (campaign / "KNOWLEDGE.json").write_text(
        json.dumps({"schema_version": 1, "snapshots": {"kda_pilot": str(index)}})
    )
    baseline = BaselineRecord(
        model_id="test/model",
        mean_e2e_s=10.0,
        workload_total_s=50.0,
        request_count=5,
        peak_memory_mib=1000.0,
        timing_scope="frozen_e2e",
        run_dir=campaign / "baseline/run",
        baseline_frames=campaign / "baseline/frames",
        sglang_commit="a" * 40,
    )
    root = Path(__file__).resolve().parents[1]
    registry = TechniqueRegistry.load(root / "techniques/registry.toml")

    contract = build_delivery_contract(
        campaign=campaign,
        technique="kernel",
        registry=registry,
        baseline=baseline,
        command_template=None,
    )
    worktree = tmp_path / "executor/worktree"
    delivery = worktree / "DELIVERY.json"
    materialized = materialize_delivery_contract(
        contract,
        worktree=worktree,
        delivery=delivery,
    )

    assert materialized["delivery_path"] == str(delivery.resolve())
    assert materialized["schemas"]["delivery"]["title"] == "Delivery"
    assert materialized["schemas"]["technique_evidence"]["title"] == (
        "KernelEvidence"
    )
    assert materialized["pinned_kernelwiki_sources"] == [
        {"path": str(reference.resolve()), "sha256": digest}
    ]
    assert materialized["performance_artifact_required_fields"]["request_count"] == 5
    assert materialized["preflight_argv"][-1] == str(delivery.resolve())
    assert materialized["gpu_inventory"] == {
        "path": str(inventory_path),
        "sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
    }
